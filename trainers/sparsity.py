import torch as t
from ..trainers.standard import StandardTrainer


class SparsityTrainer(StandardTrainer):
    """
    SAE training scheme with quadratic activations.
    """
    def __init__(self, 
                 ae, 
                 lr=1e-3, 
                 l1_penalty=1e-1,
                 warmup_steps=1000, # lr warmup period at start of training and after each resample
                 resample_steps=None, # how often to resample neurons
                 activation_func='quadratic',
                 p=0.5,
                 device=None,
    ):
        ae.set_activation_func(activation_func)
        super().__init__(
            ae,
            lr=lr,
            l1_penalty=l1_penalty,
            warmup_steps=warmup_steps,
            resample_steps=resample_steps,
            device=device)
        self.p = p

    def loss(self, x):
        x_hat, f = self.ae(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        lp_loss = f.norm(p=self.p, dim=-1).mean()

        if self.steps_since_active is not None:
            # update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        return l2_loss + self.l1_penalty * lp_loss