import numpy as np
import torch as t
from nnsight import DEFAULT_PATCHER
from nnsight.tracing.Proxy import proxy_wrapper
from nnsight.patching import Patch

"""
Implements the standard SAE training scheme.
"""

from ..dictionary import AutoEncoder
from ..trainers.trainer import SAETrainer
from ..config import DEBUG


DEFAULT_PATCHER.add(Patch(t, proxy_wrapper(t.zeros), "zeros"))


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

class ListaTrainer(SAETrainer):
    """
    Standard SAE training scheme.
    """
    def __init__(self, 
                 ae, 
                 lr=1e-3, 
                 sparsity_coefficient=0.5,
                 warmup_steps=1000, # lr warmup period at start of training and after each resample
                 resample_steps=None, # how often to resample neurons
                 device=None,
    ):
        super().__init__(ae)
        self.lr = lr
        self.sparsity_coefficient=sparsity_coefficient
        self.warmup_steps = warmup_steps

        if device is None:
            self.device = t.device('cuda' if t.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.ae.to(self.device)

        self.resample_steps = resample_steps

        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = t.zeros(ae.dict_size, dtype=int).to(self.device)
        else:
            self.steps_since_active = None 

        self.encoder_optimizer = t.optim.Adam(ae.encoder.parameters(), lr=lr)
        self.decoder_optimizer = ConstrainedAdam([ae.bias, *ae.decoder.parameters()], ae.decoder.parameters(), lr=lr)
        if resample_steps is None:
            def warmup_fn(step):
                return min(step / warmup_steps, 1.)
        else:
            def warmup_fn(step):
                return min((step % resample_steps) / warmup_steps, 1.)
        self.encoder_scheduler = t.optim.lr_scheduler.LambdaLR(self.encoder_optimizer, lr_lambda=warmup_fn)
        self.decoder_scheduler = t.optim.lr_scheduler.LambdaLR(self.decoder_optimizer, lr_lambda=warmup_fn)
        
    def compute_optimal_sparse_codes(self, x, basis, lambd, num_iter, eta=None, tol=1e-6):
        """
        Fast Iterative Shrinkage and Thresholding Algorithm (FISTA) with non-negativity
        constraint on z and early stopping.

        Parameters:
            x (tensor): Data matrix.
            basis (tensor): Basis matrix.
            lambd (float): Regularization parameter.
            num_iter (int): Maximum number of iterations.
            eta (float, optional): Step size. If None, it's computed from the largest 
                eigenvalue of basis' * basis.
            tol (float): Tolerance for early stopping based on the norm of residuals.

        Returns:
            tensor: Optimal sparse codes.
            tensor: Final residuals.
        """
        
        with t.no_grad():
        
            # optional: compute step size
            if eta is None:
                L = t.max(t.linalg.eigh(t.mm(basis, basis.T))[0])
                eta = 1./L

            # initialize momentum parameters
            tk_n = 1.
            tk = 1.
            
            # initialize optimization parameters
            x_hat_shape = (basis.size(1), x.size(1))
            residuals = t.zeros_like(x).cuda()
            ahat = t.zeros(x_hat_shape).cuda() # self.ae.encoder(x.T).cuda().T # 
            ahat_y = t.zeros(x_hat_shape).cuda()

            for i in range(num_iter):
                tk = tk_n
                tk_n = (1+np.sqrt(1+4*tk**2))/2
                ahat_pre = ahat
                residuals = x - t.mm(basis,ahat_y)
                
                # Check for early stopping condition
                # if t.norm(ahat - ahat_pre) < tol:
                #     print(f"stopped after {i} steps")
                #     break
                
                ahat_y = ahat_y.add(eta * basis.t().mm(residuals))
                ahat = ahat_y.sub(eta * lambd).clamp(min = 0.)
                ahat_y = ahat.add(ahat.sub(ahat_pre).mul((tk-1)/(tk_n)))
            residuals = x - t.mm(basis,ahat)
            return ahat.T, residuals
    
    def loss(self, x, component):
        
        if component == "decoder":
            
            # L = \norm{D(E(x)) - x}_2
            x_hat = self.ae(x, output_features=False)
            l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
            return l2_loss
            
        elif component == "encoder":
            
            # compute optimal z given D
            z, _ = self.compute_optimal_sparse_codes(
                x.T, 
                self.ae.decoder.weight, 
                self.sparsity_coefficient, 
                500  # iterations
            )
            
            # L = \norm{E(x) - z}_2
            _, f = self.ae(x, output_features=True)
            l2_loss = t.linalg.norm(f - z, dim=-1).mean()
            return l2_loss
            
        else:
            
            x_hat, f = self.ae(x, output_features=True)
            l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
            l1_loss = f.norm(p=1, dim=-1).mean()
            return l2_loss + self.l1_penalty * l1_loss

    def update(self, step, activations):
        activations = activations.to(self.device)
        
        # (1) fix E, update D for \norm{D(E(x)) - x}_2
        if step % 2 == 0:
            
            self.decoder_optimizer.zero_grad()
            loss = self.loss(activations, component="decoder")
            loss.backward()
            self.decoder_optimizer.step()
            self.decoder_scheduler.step()
        
        # (2) fix D, update E for \norm{E(x) - z}_2 where z is optimal for D
        else: 
            
            self.encoder_optimizer.zero_grad()
            loss = self.loss(activations, component="encoder")
            loss.backward()
            self.encoder_optimizer.step()
            self.encoder_scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == -1:
            self.resample_neurons(self.steps_since_active > self.resample_steps, activations)