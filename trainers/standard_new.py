"""
Implements the SAE training scheme from https://transformer-circuits.pub/2024/april-update/index.html#training-saes
"""
import torch as t
from ..trainers.trainer import SAETrainer
from ..config import DEBUG
from ..dictionary import AutoEncoderNew

class StandardTrainerNew(SAETrainer):
    """
    Standard SAE training scheme.
    """
    def __init__(self,
                 dict_class=AutoEncoderNew,
                 activation_dim=512,
                 dict_size=64*512,
                 lr=5e-5, 
                 l1_penalty=1e-1,
                 decay_start=24000, # when does the lr decay start
                 steps=30000, # when when does training end
                 seed=None,
                 device=None,
    ):
        super().__init__(seed)

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)

        self.lr = lr
        self.l1_penalty=l1_penalty

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr)
        def lr_fn(step):
            if step > steps: raise ValueError("step > steps")
            if step < decay_start:
                return 1.
            else:
                return (steps - step) / (steps - decay_start)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
    
    def loss(self, x):
        f = self.ae.encode(x)
        x_hat = self.ae.decode(f)
        # multiply f by decoder column norms
        f = f * self.ae.decoder.weight.norm(dim=0, keepdim=True)

        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        L_sparse = f.norm(p=1, dim=-1).mean()
        return L_recon + self.l1_penalty * L_sparse

    def update(self, step, x):
        x = x.to(self.device)
        x = x / x.norm(dim=-1, keepdim=True).mean() * (self.ae.activation_dim ** 0.5)

        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()

        # clip grad norm
        t.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.scheduler.step()

    @property
    def config(self):
        return {
            'trainer_class' : 'StandardTrainer',
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'warmup_steps' : self.warmup_steps,
            'device' : self.device
        }