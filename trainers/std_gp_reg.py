"""
Implements the standard SAE training scheme + uses gradient pursuit to train the encoder
"""
import torch as t
from ..trainers.standard import StandardTrainer
from ..config import DEBUG
from ..dictionary import AutoEncoder
from ..grad_pursuit import grad_pursuit



class StandardTrainerGPReg(StandardTrainer):
    """
    Standard SAE training scheme.
    """
    def __init__(self,
                 dict_class=AutoEncoder,
                 activation_dim=512,
                 dict_size=64*512,
                 lr=1e-3, 
                 l1_penalty=1e-1,
                 gp_reg_coeff=1.0, # the coefficient for grad_pursuit part of the loss
                 target_l0=25, # how many features should grad_pursuit select
                 warmup_steps=1000, # lr warmup period at start of training and after each resample
                 resample_steps=None, # how often to resample neurons
                 seed=None,
                 device=None,
                 wandb_name='StandardTrainer',
    ):
        super().__init__(
            dict_class=dict_class,
            activation_dim=activation_dim,
            dict_size=dict_size,
            lr=lr,
            l1_penalty=l1_penalty,
            warmup_steps=warmup_steps,
            resample_steps=resample_steps,
            seed=seed,
            device=device,
            wandb_name=wandb_name)

        self.gp_reg_coeff = gp_reg_coeff
        self.target_l0 = target_l0

    def loss(self, x):
        x_hat, f = self.ae(x, output_features=True)

        # compute greedy approximation of best latent code
        f_gp = grad_pursuit(
            x,
            self.ae.decoder.weight,
            target_l0=self.target_l0,
            device=self.device)

        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        gp_loss = (1 / self.target_l0)*(f - f_gp).pow(2).sum(dim=-1).mean()

        if self.steps_since_active is not None:
            # update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        return l2_loss + self.l1_penalty * l1_loss + self.gp_reg_coeff * gp_loss

    @property
    def config(self):
        return {
            'trainer_class' : 'StandardTrainer',
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'warmup_steps' : self.warmup_steps,
            'resample_steps' : self.resample_steps,
            'gp_reg_coeff' : self.gp_reg_coeff,
            'target_l0' : self.target_l0,
            'device' : self.device,
            'wandb_name': self.wandb_name,
        }