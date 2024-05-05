"""
Implements the training scheme for a gated SAE described in https://arxiv.org/abs/2404.16014
"""

import torch as t
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

class GatedAnnealTrainer(SAETrainer):
    """
    Gated SAE training scheme.
    """
    def __init__(self,
                 dict_class=GatedAutoEncoder,
                 activation_dim=512,
                 dict_size=64*512,
                 lr=3e-4, 
                 l1_penalty=1e-1,
                 warmup_steps=1000, # lr warmup period at start of training and after each resample
                 resample_steps=None, # how often to resample neurons
                 sparsity_function='Lp^p', # Lp or Lp^p
                 initial_sparsity_penalty=1e-1, # equal to l1 penalty in standard trainer
                 anneal_start=15000, # step at which to start annealing p
                 p_start=1, # starting value of p (constant throughout warmup)
                 p_end=0, # annealing p_start to p_end linearly after warmup_steps, exact endpoint excluded
                 n_sparsity_updates = 10, # number of times to update the sparsity penalty, at most steps-anneal_start times
                 sparsity_queue_length = 10, # number of recent sparsity loss terms, onle needed for adaptive_sparsity_penalty
                 resample_steps=None, # number of steps after which to resample dead neurons
                 steps=None, # total number of steps to train for
                 device=None,
                 seed=42,
                 wandb_name='GatedAnnealTrainer',
    ):
        super().__init__(seed)

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        # initialize dictionary
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.ae = dict_class(activation_dim, dict_size)
        
        if device is None:
            self.device = 'cuda' if t.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.ae.to(self.device)
                
        self.lr = lr
        self.sparsity_function = sparsity_function
        self.anneal_start = anneal_start
        self.p_start = p_start
        self.p_end = p_end
        self.p = p_start # p is set in self.loss()
        self.next_p = None # set in self.loss()
        self.lp_loss = None # set in self.loss()
        self.scaled_lp_loss = None # set in self.loss()
        self.n_sparsity_updates = n_sparsity_updates
        self.sparsity_update_steps = t.linspace(anneal_start, steps, n_sparsity_updates, dtype=int)
        self.p_values = t.linspace(p_start, p_end, n_sparsity_updates)
        self.p_step_count = 0
        self.current_sparsity_penalty = initial_sparsity_penalty # alpha
        self.sparsity_queue_length = sparsity_queue_length
        self.sparsity_queue = []

        self.warmup_steps = warmup_steps
        self.steps = steps
        self.logging_parameters = ['p', 'next_p', 'lp_loss', 'scaled_lp_loss', 'current_sparsity_penalty']
        self.seed = seed
        self.wandb_name = wandb_name

        self.resample_steps = resample_steps
        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = t.zeros(self.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None 

        self.optimizer = ConstrainedAdam(self.ae.parameters(), self.ae.decoder.parameters(), lr=lr)
        if resample_steps is None:
            def warmup_fn(step):
                return min(step / warmup_steps, 1.)
        else:
            def warmup_fn(step):
                return min((step % resample_steps) / warmup_steps, 1.)
        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_fn)
        
    def loss(self, x, logging=False, **kwargs):
        x_hat = self.ae(x)
        f_gate = self.ae.encode(x, gate_only=True)
        x_hat_gate = f_gate @ self.ae.decoder.weight.detach().T + self.ae.decoder_bias.detach()

        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        L_sparse = t.linalg.norm(f_gate, ord=1, dim=-1).mean()
        L_aux = (x - x_hat_gate).pow(2).sum(dim=-1).mean()

        loss = L_recon + self.l1_penalty * L_sparse + L_aux

        
            
            
        # Compute loss terms
        f = f_gate
        self.lp_loss = self.lp_norm(f, self.p)

        if self.next_p is not None:
            lp_loss_next = self.lp_norm(f, self.next_p)
            self.sparsity_queue.append([self.lp_loss.item(), lp_loss_next.item()])
            self.sparsity_queue = self.sparsity_queue[-self.sparsity_queue_length:]
    
        if step in self.sparsity_update_steps:
            # Adapt sparsity penalty alpha
            if self.next_p is not None:
                local_sparsity_new = t.tensor([i[0] for i in self.sparsity_queue]).mean()
                local_sparsity_old = t.tensor([i[1] for i in self.sparsity_queue]).mean()
                self.current_sparsity_penalty = self.current_sparsity_penalty * (local_sparsity_new / local_sparsity_old).item()
            # Update p
            self.p = self.p_values[self.p_step_count].item()
            if self.p_step_count < self.n_sparsity_updates:
                self.next_p = self.p_values[self.p_step_count+1].item()
            self.p_step_count += 1

        # Update dead feature count
        if self.steps_since_active is not None:
            # update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        self.scaled_lp_loss = self.lp_loss * self.current_sparsity_penalty
        return l2_loss + self.scaled_lp_loss
    
    
    
        if not logging:
            return loss
        else:
            return namedtuple('LossLog', ['x', 'x_hat', 'f', 'losses'])(
                x, x_hat, self.ae.encode(x),
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
        loss = self.loss(x)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

    @property
    def config(self):
        return {
            'trainer_class' : 'GatedSAETrainer',
            'activation_dim' : self.ae.activation_dim,
            'dict_size' : self.ae.dict_size,
            'lr' : self.lr,
            'l1_penalty' : self.l1_penalty,
            'warmup_steps' : self.warmup_steps,
            'device' : self.device,
            'wandb_name': self.wandb_name,
        }