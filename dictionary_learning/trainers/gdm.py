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


class GatedSAETrainer(SAETrainer):
    """
    Gated SAE training scheme.
    """

    def __init__(
        self,
        dict_class=GatedAutoEncoder,
        activation_dim=512,
        dict_size=64 * 512,
        lr=5e-5,
        l1_penalty=1e-1,
        warmup_steps=1000,  # lr warmup period at start of training and after each resample
        resample_steps=None,  # how often to resample neurons
        seed=None,
        device=None,
        layer=None,
        lm_name=None,
        wandb_name="GatedSAETrainer",
        submodule_name=None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name

        if seed is not None:
            t.manual_seed(seed)
            t.cuda.manual_seed_all(seed)

        # initialize dictionary
        self.ae = dict_class(activation_dim, dict_size)

        self.lr = lr
        self.l1_penalty = l1_penalty
        self.warmup_steps = warmup_steps
        self.wandb_name = wandb_name

        if device is None:
            self.device = "cuda" if t.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        self.optimizer = ConstrainedAdam(
            self.ae.parameters(), self.ae.decoder.parameters(), lr=lr
        )

        def warmup_fn(step):
            return min(1, step / warmup_steps)

        self.scheduler = t.optim.lr_scheduler.LambdaLR(self.optimizer, warmup_fn)

    def loss(self, x, logging=False, **kwargs):
        f, f_gate = self.ae.encode(x, return_gate=True)
        x_hat = self.ae.decode(f)
        x_hat_gate = (
            f_gate @ self.ae.decoder.weight.detach().T + self.ae.decoder_bias.detach()
        )

        L_recon = (x - x_hat).pow(2).sum(dim=-1).mean()
        L_sparse = t.linalg.norm(f_gate, ord=1, dim=-1).mean()
        L_aux = (x - x_hat_gate).pow(2).sum(dim=-1).mean()

        loss = L_recon + self.l1_penalty * L_sparse + L_aux

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "mse_loss": L_recon.item(),
                    "sparsity_loss": L_sparse.item(),
                    "aux_loss": L_aux.item(),
                    "loss": loss.item(),
                },
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
            "dict_class": "GatedAutoEncoder",
            "trainer_class": "GatedSAETrainer",
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "lr": self.lr,
            "l1_penalty": self.l1_penalty,
            "warmup_steps": self.warmup_steps,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }
