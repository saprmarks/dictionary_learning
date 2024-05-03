import torch as t

"""
Implements the standard SAE training scheme.
"""

from ..dictionary import AutoEncoder
from ..trainers.trainer import SAETrainer
from ..config import DEBUG

class ConstrainedAdam(t.optim.Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    """
    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr)
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

class PAnnealTrainer(SAETrainer):
    """
    SAE training scheme with the option to anneal the sparsity parameter p.
    You can further choose to use Lp or Lp^p sparsity.
    """
    def __init__(self, 
                 dict_class=AutoEncoder,
                 activation_dim=512,
                 dict_size=64*512, 
                 lr=1e-3, 
                 warmup_steps=1000, # lr warmup period at start of training and after each resample
                 sparsity_function='Lp', # Lp or Lp^p
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
                 wandb_name='PAnnealTrainer',
    ):
        super().__init__(seed)

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        if device is None:
            self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # initialize dictionary
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.ae = dict_class(activation_dim, dict_size)
        self.ae.to(self.device)
        
        self.lr = lr
        self.sparsity_function = sparsity_function
        self.anneal_start = anneal_start
        self.p_start = p_start
        self.p_end = p_end
        self.p = p_start # p is set in self.loss()
        self.next_p = p_start
        self.lp_loss = None # lp_loss is set in self.loss()
        self.n_sparsity_updates = n_sparsity_updates
        self.sparsity_update_steps = t.linspace(anneal_start, steps, n_sparsity_updates+1, dtype=int)
        self.p_values = t.linspace(p_start, p_end, n_sparsity_updates+1)
        self.initial_sparsity_penalty = initial_sparsity_penalty
        self.current_sparsity_penalty = initial_sparsity_penalty # alpha
        self.sparsity_queue_length = sparsity_queue_length
        self.sparsity_queue = t.zeros(0, 2).to(self.device)

        self.warmup_steps = warmup_steps
        self.steps = steps
        self.logging_parameters = ['p', 'lp_loss', 'current_sparsity_penalty']
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

    def resample_neurons(self, deads, activations):
        with t.no_grad():
            if deads.sum() == 0: return
            print(f"resampling {deads.sum().item()} neurons")

            # compute loss for each activation
            losses = (activations - self.ae(activations)).norm(dim=-1)

            # sample input to create encoder/decoder weights from
            n_resample = min([deads.sum(), losses.shape[0]])
            indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
            sampled_vecs = activations[indices]

            # reset encoder/decoder weights for dead neurons
            alive_norm = self.ae.encoder.weight[~deads].norm(dim=-1).mean()
            self.ae.encoder.weight[deads][:n_resample] = sampled_vecs * alive_norm * 0.2
            self.ae.decoder.weight[:,deads][:,:n_resample] = (sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)).T
            self.ae.encoder.bias[deads][:n_resample] = 0.


            # reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()['state']
            ## encoder weight
            state_dict[1]['exp_avg'][deads] = 0.
            state_dict[1]['exp_avg_sq'][deads] = 0.
            ## encoder bias
            state_dict[2]['exp_avg'][deads] = 0.
            state_dict[2]['exp_avg_sq'][deads] = 0.
            ## decoder weight
            state_dict[3]['exp_avg'][:,deads] = 0.
            state_dict[3]['exp_avg_sq'][:,deads] = 0.

    def lp_norm(self, f, p):
        if self.sparsity_function == 'Lp':
            return t.norm(f, p=p, dim=-1).mean()
        elif self.sparsity_function == 'Lp^p':
            return t.norm(f, p=p, dim=-1).pow(p).mean()
    
    def loss(self, x, step):
        # Compute loss terms
        x_hat, f = self.ae(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        self.lp_loss = self.lp_norm(f, self.p)
        lp_loss_next = self.lp_norm(f, self.next_p)
        lp_losses = t.hstack([self.lp_loss, lp_loss_next]).unsqueeze(dim=0)

        # Keep a buffer of recent feature activations for determining sparsity penalty alpha
        self.sparsity_queue = t.vstack([self.sparsity_queue, lp_losses])
        if self.sparsity_queue.shape[0] > self.sparsity_queue_length:
            self.sparsity_queue = self.sparsity_queue[1:]
            print(f'sparsity_queue.shape: {self.sparsity_queue.shape}')
   
        # Adapt sparsity penalty alpha
        if step in self.sparsity_update_steps:
            # relative_progress = max(step - self.anneal_start, 0) / (self.steps - self.anneal_start)
            # p_new = self.p_start + relative_progress * (self.p_end - self.p_start)
            local_sparsity_new = self.sparsity_queue[0, :].mean()
            local_sparsity_old = self.sparsity_queue[1, :].mean()
            self.current_sparsity_penalty = self.initial_sparsity_penalty * (local_sparsity_new / local_sparsity_old).item()
            self.p = self.p_values[0].item()
            if self.sparsity_queue.shape[0] > 1:
                self.next_p = self.p_values[1].item()
            self.sparsity_queue = self.sparsity_queue[1:]

        # Update dead feature count
        if self.steps_since_active is not None:
            # update steps_since_active
            deads = (f == 0).all(dim=0)
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0
        
        return l2_loss + self.current_sparsity_penalty * self.lp_loss
    
        
    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations, step)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == self.resample_steps - 1:
            self.resample_neurons(self.steps_since_active > self.resample_steps / 2, activations)

    @property
    def config(self):
        return {
            'trainer_class' : "PAnnealTrainer",
            'dict_class' : "AutoEncoder",
            'activation_dim' : self.activation_dim,
            'dict_size' : self.dict_size,
            'lr' : self.lr,
            'sparsity_function' : self.sparsity_function,
            'sparsity_penalty' : self.current_sparsity_penalty,
            'p_start' : self.p_start,
            'p_end' : self.p_end,
            'anneal_start' : self.anneal_start,
            'sparsity_queue_length' : self.sparsity_queue_length,
            'n_sparsity_updates' : self.n_sparsity_updates,
            'warmup_steps' : self.warmup_steps,
            'resample_steps' : self.resample_steps,
            'steps' : self.steps,
            'seed' : self.seed,
            'wandb_name' : self.wandb_name,
        }