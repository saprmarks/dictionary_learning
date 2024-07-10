"""
Implements the SAE training scheme from https://arxiv.org/abs/2406.04093.
Significant portions of this code have been copied from https://github.com/EleutherAI/sae/blob/main/sae
"""

import einops
import torch as t
import torch.nn as nn
from collections import namedtuple

from ..config import DEBUG
from ..dictionary import Dictionary
from ..kernels import TritonDecoder
from ..trainers.trainer import SAETrainer


@t.no_grad()
def geometric_median(points: t.Tensor, max_iter: int = 100, tol: float = 1e-5):
    """Compute the geometric median `points`. Used for initializing decoder bias."""
    # Initialize our guess as the mean of the points
    guess = points.mean(dim=0)
    prev = t.zeros_like(guess)

    # Weights for iteratively reweighted least squares
    weights = t.ones(len(points), device=points.device)

    for _ in range(max_iter):
        prev = guess

        # Compute the weights
        weights = 1 / t.norm(points - guess, dim=1)

        # Normalize the weights
        weights /= weights.sum()

        # Compute the new geometric median
        guess = (weights.unsqueeze(1) * points).sum(dim=0)

        # Early stopping condition
        if t.norm(guess - prev) < tol:
            break

    return guess


class AutoEncoderTopK(Dictionary, nn.Module):
    """
    The top-k autoencoder architecture and initialization used in https://arxiv.org/abs/2406.04093
    """
    def __init__(self, activation_dim, dict_size, k):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k
        
        self.encoder = nn.Linear(activation_dim, dict_size)
        self.encoder.bias.data.zero_()
        
        self.decoder = nn.Parameter(self.encoder.weight.data.clone())
        self.set_decoder_norm_to_unit_norm()
        
        self.b_dec = self.b_dec = nn.Parameter(t.zeros(activation_dim))

    def encode(self, x):
        return nn.functional.relu(self.encoder(x - self.b_dec))
    
    def decode(self, top_acts, top_indices):
        d = TritonDecoder.apply(top_indices, top_acts, self.decoder.mT)
        return d + self.b_dec
    
    def forward(self, x, output_features=False):
        # (rangell): some shape hacking going on here
        f = self.encode(x.view(-1, x.shape[-1]))
        top_acts, top_indices = f.topk(self.k, sorted=False)
        x_hat = self.decode(top_acts, top_indices).view(x.shape)
        f = f.view(*x.shape[:-1], f.shape[-1])
        if not output_features:
            return x_hat
        else:
            return x_hat, f
        
    @t.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = t.finfo(self.decoder.dtype).eps
        norm = t.norm(self.decoder.data, dim=1, keepdim=True)
        self.decoder.data /= norm + eps
        
    @t.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):
        assert self.decoder.grad is not None  # keep pyright happy

        parallel_component = einops.einsum(
            self.decoder.grad,
            self.decoder.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        self.decoder.grad -= einops.einsum(
            parallel_component,
            self.decoder.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
                   
    def from_pretrained(path, k=100, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict['encoder.weight'].shape
        autoencoder = AutoEncoderTopK(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder
    

class TrainerTopK(SAETrainer):
    """
    Top-K SAE training scheme.
    """
    def __init__(self,
                 dict_class=AutoEncoderTopK,
                 activation_dim=512,
                 dict_size=64*512,
                 k=100,
                 auxk_alpha=1/32,  # see Appendix A.2
                 decay_start=24000, # when does the lr decay start
                 steps=30000, # when when does training end
                 seed=None,
                 device=None,
                 layer=None,
                 lm_name=None,
                 wandb_name='AutoEncoderTopK',
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name

        self.wandb_name = wandb_name
        self.steps = steps
        self.k = k
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # Initialise autoencoder
        self.ae = dict_class(activation_dim, dict_size, k)
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)
        
        # Auto-select LR using 1 / sqrt(d) scaling law from Figure 3 of the paper
        scale = dict_size / (2 ** 14)
        self.lr = 2e-4 / scale ** 0.5
        self.auxk_alpha = auxk_alpha
        self.dead_feature_threshold = 10_000_000
        
        # Optimizer and scheduler
        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=self.lr, betas=(0.9, 0.999))
        def lr_fn(step):
            if step < decay_start:
                return 1.
            else:
                return (steps - step) / (steps - decay_start)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
    
        # Training parameters
        self.num_tokens_since_fired = t.zeros(dict_size, dtype=t.long, device=device)
        
        # Log the effective L0, i.e. number of features actually used, which should a constant value (K)
        # Note: The standard L0 is essentially a measure of dead features for Top-K SAEs)
        self.logging_parameters = ["effective_l0", "dead_features"]
        self.effective_l0 = -1
        self.dead_features = -1
        
    def loss(self, x, step=None, logging=False):
        
        # Run the SAE
        f = self.ae.encode(x)
        top_acts, top_indices = f.topk(self.k, sorted=False)
        x_hat = self.ae.decode(top_acts, top_indices)
        
        # Measure goodness of reconstruction
        e = x_hat - x
        total_variance = (x - x.mean(0)).pow(2).sum(0)
        
        # Update the effective L0 (again, should just be K)
        self.effective_l0 = top_acts.size(1)
        
        # Update "number of tokens since fired" for each features
        num_tokens_in_step = x.size(0)
        did_fire = t.zeros_like(self.num_tokens_since_fired, dtype=t.bool)
        did_fire[top_indices.flatten()] = True
        self.num_tokens_since_fired += num_tokens_in_step
        self.num_tokens_since_fired[did_fire] = 0
        
        # Compute dead feature mask based on "number of tokens since fired"
        dead_mask = (
            self.num_tokens_since_fired > self.dead_feature_threshold
            if self.auxk_alpha > 0
            else None
        ).to(f.device)
        self.dead_features = int(dead_mask.sum())
        
        # If dead features: Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            
            # Heuristic from Appendix B.1 in the paper
            k_aux = x.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = t.where(dead_mask[None], f, -t.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.ae.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).pow(2) #.sum(0)
            auxk_loss = scale * t.mean(auxk_loss / total_variance)
        else:
            auxk_loss = x_hat.new_tensor(0.0)

        l2_loss = e.pow(2).sum(dim=-1).mean()
        auxk_loss = auxk_loss.sum(dim=-1).mean()
        loss = l2_loss + self.auxk_alpha + auxk_loss
        
        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'l2_loss': l2_loss.item(),
                    'auxk_loss': auxk_loss.item(),
                    'loss' : loss.item()
                }
            )

    def update(self, step, x):
        
        # Initialise the decoder bias
        if step == 0:
            median = geometric_median(x)
            self.ae.b_dec.data = median
            
        # Make sure the decoder is still unit-norm
        self.ae.set_decoder_norm_to_unit_norm()
        
        # compute the loss
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        # clip grad norm and remove grads parallel to decoder directions
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)
        self.ae.remove_gradient_parallel_to_decoder_directions()

        # do a training step
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.scheduler.step()
        return loss.item()

    @property
    def config(self):
        return {
            'trainer_class' : 'TrainerTopK',
            'dict_class' : 'AutoEncoderTopK',
            'lr' : self.lr,
            'steps' : self.steps,
            'seed' : self.seed,
            'activation_dim' : self.ae.activation_dim,
            'dict_size' : self.ae.dict_size,
            'k': self.ae.k,
            'device' : self.device,
            "layer" : self.layer,
            'lm_name' : self.lm_name,
            'wandb_name' : self.wandb_name,
        }
