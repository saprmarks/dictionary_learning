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

    def loss(self, x):
        x_hat = self.ae(x)
        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        return L_recon
    
    def update(self, step, x):
        x = x.to(self.device)
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

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
        }