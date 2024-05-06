"""
Combines gated SAEs with Anthropic's training scheme from their April update
"""
import torch as t
import torch.nn.init as init
from ..trainers.trainer import SAETrainer
from ..config import DEBUG
from ..dictionary import GatedAutoEncoder
from collections import namedtuple


def initialize(ae):
    """
    Initialization scheme for the autoencoder.
    """
    init.zeros_(ae.decoder_bias)
    init.zeros_(ae.r_mag)
    init.zeros_(ae.gate_bias)
    init.zeros_(ae.mag_bias)

    w = t.randn_like(ae.encoder.weight)
    w = w / w.norm(dim=0, keepdim=True) * 0.1
    ae.encoder.weight = t.nn.Parameter(w)
    ae.decoder.weight = t.nn.Parameter(w.T)


class GatedTrainerNew(SAETrainer):
    """
    Standard SAE training scheme.
    """
    def __init__(self,
                 dict_class=GatedAutoEncoder,
                 activation_dim=512,
                 dict_size=64*512,
                 lr=5e-4, 
                 l1_penalty=1e-1,
                 lambda_warm_steps=1500, # steps over which to warm up the l1 penalty
                 decay_start=24000, # when does the lr decay start
                 steps=30000, # when when does training end
                 seed=None,
                 device=None,
                 wandb_name='GatedTrainerNew',
    ):
        super().__init__(seed)

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size, initialization=initialize, device=self.device)

        self.lr = lr
        self.l1_penalty=l1_penalty
        self.lambda_warm_steps=lambda_warm_steps
        self.decay_start=decay_start
        self.steps = steps
        self.wandb_name = wandb_name

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr)
        def lr_fn(step):
            if step < decay_start:
                return 1.
            else:
                return (steps - step) / (steps - decay_start)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
    
    def loss(self, x, step=None, logging=False):
        f, f_gate = self.ae.encode(x, return_gate=True)
        x_hat = self.ae.decode(f)
        x_hat_gate = f_gate @ self.ae.decoder.weight.detach().T + self.ae.decoder_bias.detach()

        f = f * self.ae.decoder.weight.norm(dim=0, keepdim=True)

        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        L_sparse = f_gate.norm(p=1, dim=-1).mean()
        L_aux = (x - x_hat_gate).pow(2).sum(dim=-1).mean()

        l1_penalty = self.l1_penalty
        l1_penalty = min(1., step / self.lambda_warm_steps) * self.l1_penalty

        loss = L_recon + l1_penalty * L_sparse + L_aux
        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'mse_loss' : L_recon.item(),
                    'sparsity_loss' : L_sparse.item(),
                    'aux_loss' : L_aux.item(),
                    'l1_penalty' : l1_penalty,
                    'loss' : loss.item()
                }
            )
    

    def update(self, step, x):
        x = x.to(self.device)
        
        # normalization was removed
        # x = x / x.norm(dim=-1).mean() * (self.ae.activation_dim ** 0.5)

        self.optimizer.zero_grad()
        loss = self.loss(x, step=step)
        loss.backward()

        # clip grad norm
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()

        return loss.item()

    @property
    def config(self):
        return {
            'trainer_class' : 'GatedTrainerNew',
            'dict_class' : 'GatedAutoEncoder',
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'lambda_warm_steps' : self.lambda_warm_steps,
            'decay_start' : self.decay_start,
            'steps' : self.steps,
            'seed' : self.seed,
            'activation_dim' : self.ae.activation_dim,
            'dict_size' : self.ae.dict_size,
            'device' : self.device,
            'wandb_name' : self.wandb_name,
        }