"""
Implements the training scheme for a gated SAE described in https://arxiv.org/abs/2404.16014
"""

import torch as t
from typing import Optional

from ..trainers.trainer import SAETrainer
from ..config import DEBUG
from ..dictionary import GatedAutoEncoder
from collections import namedtuple

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

class GatedSAETrainer(SAETrainer):
    """
    Gated SAE training scheme.
    """
    def __init__(self,
                 steps: int, # total number of steps to train for
                 activation_dim: int,
                 dict_size: int,
                 layer: int,
                 lm_name: str,
                 dict_class = GatedAutoEncoder,
                 lr: float = 5e-5,
                 l1_penalty: float = 1e-1,
                 warmup_steps: int = 1000, # lr warmup period at start of training and after each resample
                 sparsity_warmup_steps: Optional[int] = 2000,
                 decay_start:Optional[int]=None, # decay learning rate after this many steps
                 seed: Optional[int] = None,
                 device: Optional[str] = None,
                 wandb_name: Optional[str] = 'GatedSAETrainer',
                 submodule_name: Optional[str] = None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)

        self.lr = lr
        self.l1_penalty=l1_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.decay_start = decay_start
        self.wandb_name = wandb_name

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

        if decay_start is not None:
            assert 0 <= decay_start < steps, "decay_start must be >= 0 and < steps."
            assert decay_start > warmup_steps, "decay_start must be > warmup_steps."
            if sparsity_warmup_steps is not None:
                assert decay_start > sparsity_warmup_steps, "decay_start must be > sparsity_warmup_steps."

        assert 0 <= warmup_steps < steps, "warmup_steps must be >= 0 and < steps."

        if sparsity_warmup_steps is not None:
            assert 0 <= sparsity_warmup_steps < steps, "sparsity_warmup_steps must be >= 0 and < steps."

        def warmup_fn(step):
            if step < warmup_steps:
                return step / warmup_steps
            
            if decay_start is not None and step >= decay_start:
                return (steps - step) / (steps - decay_start)

            return 1.0

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_fn)

    def loss(self, x:t.Tensor, step:int, logging:bool=False, **kwargs):

        if self.sparsity_warmup_steps is not None:
            sparsity_scale = min(step / self.sparsity_warmup_steps, 1.0)
        else:
            sparsity_scale = 1.0

        f, f_gate = self.ae.encode(x, return_gate=True)
        x_hat = self.ae.decode(f)
        x_hat_gate = f_gate @ self.ae.decoder.weight.detach().T + self.ae.decoder_bias.detach()

        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        L_sparse = t.linalg.norm(f_gate, ord=1, dim=-1).mean()
        L_aux = (x - x_hat_gate).pow(2).sum(dim=-1).mean()

        loss = L_recon + (self.l1_penalty * L_sparse * sparsity_scale) + L_aux

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'mse_loss' : L_recon.item(),
                    'sparsity_loss' : L_sparse.item(),
                    'aux_loss' : L_aux.item(),
                    'loss' : loss.item()
                }
            )
    
    def update(self, step, x):
        x = x.to(self.device)
        self.optimizer.zero_grad()
        loss = self.loss(x, step)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    @property
    def config(self):
        return {
            'dict_class': 'GatedAutoEncoder',
            'trainer_class' : 'GatedSAETrainer',
            'activation_dim' : self.ae.activation_dim,
            'dict_size' : self.ae.dict_size,
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'warmup_steps' : self.warmup_steps,
            'sparsity_warmup_steps' : self.sparsity_warmup_steps,
            'decay_start' : self.decay_start,
            'seed' : self.seed,
            'device' : self.device,
            'layer' : self.layer,
            'lm_name' : self.lm_name,
            'wandb_name': self.wandb_name,
            'submodule_name': self.submodule_name,
        }
