import torch as t

"""
Implements the standard SAE training scheme.
"""

from ..trainers.standard import StandardTrainer
from ..config import DEBUG


class GroupSAETrainer(StandardTrainer):
    
    def __init__(self, ae, group_size, lr=0.001, l1_penalty=0.1, warmup_steps=1000, resample_steps=None, device=None):
        super().__init__(ae, lr, l1_penalty, warmup_steps, resample_steps, device)
        
        self.group_size = group_size
        self.n_groups = self.ae.dict_size // self.group_size
        
    def loss(self, x):
        
        # compute features and reconstructions
        x_hat, f = self.ae(x, output_features=True)
        
        # reconstruction loss (L2 norm)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        
        # group sparsity (L1/L2 norm)
        m, _ = f.size()
        f_grouped = f.view(m, self.n_groups, self.group_size) 
        l2_norms = t.norm(f_grouped, p=2, dim=2)  # L2 norm for each group
        l1_of_l2 = t.sum(t.abs(l2_norms), dim=1)  # L1 norm across L2 norms
        reg_loss = l1_of_l2.mean()  # mean across batch
        return l2_loss + self.l1_penalty * reg_loss
