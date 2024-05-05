"""
Implements training decoder only with gradient pursuit as encoder
"""

import torch as t
from ..trainers.trainer import SAETrainer
from ..config import DEBUG
from ..dictionary import GradPursuitAutoEncoder

class ConstrainedAdam(t.optim.Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    """
    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr, betas=(0, 0.999))
        self.constrained_params = list(constrained_params)
    
    def step(self, closure=None):
        with t.no_grad():
            for p in self.constrained_params:
                normed_p = p / p.norm(dim=0, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with t.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=0, keepdim=True)

class DecoderOnlySAETrainer(SAETrainer):
    """
    Trains only decoder when using a parameterless SAE.
    """
    def __init__(self,
                 dict_class=GradPursuitAutoEncoder,
                 activation_dim=512,
                 dict_size=64*512,
                 lr=3e-4, 
                 l1_penalty=1e-1,
                 warmup_steps=1000, # lr warmup period at start of training and after each resample
                 resample_steps=None, # how often to resample neurons
                 target_l0=20,
                 seed=None,
                 device=None,
                 wandb_name="decoder_only"
    ):
        super().__init__(seed)

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size, target_l0=target_l0, device=device)

        self.lr = lr
        self.l1_penalty=l1_penalty
        self.warmup_steps = warmup_steps
        self.target_l0 = target_l0
        self.wandb_name = wandb_name
        self.resample_steps = resample_steps

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)

        self.optimizer = ConstrainedAdam(
            self.ae.parameters(),
            self.ae.decoder.parameters(),
            lr=lr
        )
        def warmup_fn(step):
            return min(1, step / warmup_steps)
        # TODO: maybe decrease learning rate with a factor 1/t
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_fn)

    def resample_neurons(self, deads, activations):
        with t.no_grad():
            if deads.sum() == 0: return
            print(f"resampling {deads.sum().item()} neurons")

            # compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # get norm of the living neurons
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()

            # resample first n_resample dead neurons
            deads[deads.nonzero()[n_resample:]] = False
            self.ae.encoder.weight[deads] = sampled_vecs * alive_norm * 0.2
            self.ae.decoder.weight[:,deads] = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.encoder.bias[deads] = 0.


            # reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()['state']
            ## encoder weight
            state_dict[1]['exp_avg'][deads] = 0.
            state_dict[1]['exp_avg_sq'][deads] = 0.
            ## encoder bias
            state_dict[2]['exp_avg'][deads] = 0.
            state_dict[2]['exp_avg_sq'][deads] = 0.
            ## decoder weight
            state_dict[3]['exp_avg'][:,deads] = 0.
            state_dict[3]['exp_avg_sq'][:,deads] = 0.

    def loss(self, x):
        x_hat = self.ae(x)
        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        return L_recon

    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    @property
    def config(self):
        return {
            'trainer_class' : 'GradPursuitSAETrainer',
            'activation_dim' : self.ae.activation_dim,
            'dict_size' : self.ae.dict_size,
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'warmup_steps' : self.warmup_steps,
            'target_l0' : self.target_l0,
            'device' : self.device,
            'wandb_name' : self.wandb_name
        }