"""
Implements the SAE training scheme from https://transformer-circuits.pub/2024/april-update/index.html#training-saes
+ uses gradient pursuit to train the encoder
"""

import torch as t
from ..trainers.standard_new import StandardTrainerNew
from ..config import DEBUG
from ..dictionary import AutoEncoderNew
from ..grad_pursuit import grad_pursuit

class StandardTrainerNewGPReg(StandardTrainerNew):
    """
    Standard SAE training scheme.
    """
    def __init__(self,
                 dict_class=AutoEncoderNew,
                 activation_dim=512,
                 dict_size=64*512,
                 lr=5e-5, 
                 l1_penalty=1e-1,
                 lambda_warm_steps=1500, # steps over which to warm up the l1 penalty
                 decay_start=24000, # when does the lr decay start
                 steps=30000, # when when does training end
                 gp_reg_coeff=1.0, # the coefficient for grad_pursuit part of the loss
                 target_l0=25, # how many features should grad_pursuit select
                 seed=None,
                 device=None,
                 wandb_name='StandardTrainerNew_Anthropic_GP_Reg',
    ):
        super().__init__(
            dict_class=dict_class,
            activation_dim=activation_dim,
            dict_size=dict_size,
            lr=lr,
            l1_penalty=l1_penalty,
            lambda_warm_steps=lambda_warm_steps,
            decay_start=decay_start,
            steps=steps,
            seed=seed,
            device=device,
            wandb_name=wandb_name)

        self.gp_reg_coeff = gp_reg_coeff
        self.target_l0 = target_l0

    def loss(self, step, x):
        x_hat, f = self.ae(x, output_features=True)
        f_gp = grad_pursuit(x, self.decoder.weight, target_l0=self.target_l0, device=self.device)

        l1_penalty = min(1., step / self.lambda_warm_steps) * self.l1_penalty

        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        L_sparse = f.norm(p=1, dim=-1).mean()
        L_gp = (f - f_gp).pow(2).sum(dim=-1).mean()
        return L_recon + l1_penalty * L_sparse + self.gp_reg_coeff * L_gp

    @property
    def config(self):
        return {
            'trainer_class' : 'StandardTrainerNewGPReg',
            'dict_class' : 'AutoEncoderNew',
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'lambda_warm_steps' : self.lambda_warm_steps,
            'decay_start' : self.decay_start,
            'steps' : self.steps,
            'seed' : self.seed,
            'activation_dim' : self.ae.activation_dim,
            'dict_size' : self.ae.dict_size,
            'gp_reg_coeff' : self.gp_reg_coeff,
            'target_l0' : self.target_l0,
            'device' : self.device,
            'wandb_name' : self.wandb_name,
        }