"""
Implements the SAE training scheme from https://transformer-circuits.pub/2024/april-update/index.html#training-saes
"""
import torch as t
from ..trainers.trainer import SAETrainer
from ..config import DEBUG
from ..dictionary import AutoEncoderNew
from collections import namedtuple

class StandardTrainerNew(SAETrainer):
    """
    Standard SAE training scheme.
    """
    def __init__(self,
                 dict_class=AutoEncoderNew,
                 activation_dim=512,
                 dict_size=64*512,
                 lr=5e-4, 
                 l1_penalty=1e-1,
                 lambda_warm_steps=1500, # steps over which to warm up the l1 penalty
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
        self.lambda_warm_steps=lambda_warm_steps
        self.decay_start=decay_start
        self.steps = steps

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)

        self.optimizer = t.optim.Adam(self.ae.parameters(), lr=lr)
        def lr_fn(step):
            if step < decay_start:
                return 1.
            else:
                return (steps - step) / (steps - decay_start)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_fn)
    
    def loss(self, x, step=None, logging=False):
        x_hat, f = self.ae(x, output_features=True)

        l1_penalty = self.l1_penalty
        l1_penalty = min(1., step / self.lambda_warm_steps) * self.l1_penalty

        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        L_sparse = f.norm(p=1, dim=-1).mean()

        loss = L_recon + l1_penalty * L_sparse

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                {
                    'mse_loss' : L_recon.item(),
                    'sparsity_loss' : L_sparse.item(),
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
            'trainer_class' : 'StandardTrainerNew',
            'dict_class' : 'AutoEncoderNew',
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'lambda_warm_steps' : self.lambda_warm_steps,
            'decay_start' : self.decay_start,
            'steps' : self.steps,
            'seed' : self.seed,
            'activation_dim' : self.ae.activation_dim,
            'dict_size' : self.ae.dict_size,
            'device' : self.device
        }