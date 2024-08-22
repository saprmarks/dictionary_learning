import torch as t
import torch.autograd as autograd
from ..trainers.trainer import SAETrainer
from ..dictionary import JumpReluAutoEncoder, StepFunction

from ..config import DEBUG
from collections import namedtuple

class JumpReluTrainer(SAETrainer):
    def __init__(self,
                 dict_class=JumpReluAutoEncoder,
                 activation_dim=512,
                 dict_size=64*512,
                 lr=5e-5, 
                 l0_penalty=1e-1,
                 warmup_steps=1000, # lr warmup period at start of training and after each resample
                 seed=None,
                 device=None,
                 layer=None,
                 lm_name=None,
                 wandb_name='JumpReluTrainer',
                 submodule_name=None,
                 set_linear_to_constant=False,
                 pre_encoder_bias=True,
    ):
        super().__init__(seed)
        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.wandb_name = wandb_name

        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.ae = dict_class(activation_dim, dict_size, device=self.device, pre_encoder_bias=pre_encoder_bias)
        self.ae.to(self.device)
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.l0_penalty = l0_penalty
        self.set_linear_to_constant = set_linear_to_constant

        self.optimizer = t.optim.Adam(self.ae.parameters(), betas=(0.9, 0.999), eps=1e-8)
        def warmup_fn(step):
            return min(1, step / warmup_steps)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_fn)

    def loss(self, x, logging=False, **kwargs):
        x_hat, f = self.ae(x, output_features=True, set_linear_to_constant=self.set_linear_to_constant)
        L_recon = (x - x_hat).pow(2).mean()
        L_spars = StepFunction.apply(f, self.ae.jump_relu.log_threshold, self.ae.bandwidth).sum(dim=-1).mean()

        loss = L_recon + self.l0_penalty * L_spars

        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, f,
                losses={
                    'mse_loss' : L_recon.item(),
                    'sparsity_loss' : L_spars.item(),
                    'loss' : loss.item()
                }
            )
        
    def update(self, step, x):
        x = x.to(self.device)
        self.optimizer.zero_grad()
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()

    @property
    def config(self):
        return {
            'dict_class': 'JumpReluAutoEncoder',
            'trainer_class': 'JumpReluTrainer',
            'activation_dim': self.ae.activation_dim,
            'dict_size': self.ae.dict_size,
            'lr': self.lr,
            'warmup_steps': self.warmup_steps,
            'l0_penalty': self.l0_penalty,
            'device': self.device,
            'layer': self.layer,
            'lm_name': self.lm_name,
            'submodule_name': self.submodule_name,
            'wandb_name': self.wandb_name,
        }