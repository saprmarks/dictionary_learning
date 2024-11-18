"""
Implements the standard SAE training scheme.
"""

import torch as th
from ..trainers.trainer import SAETrainer
from ..config import DEBUG
from ..dictionary import CrossCoder
from collections import namedtuple


class CrossCoderTrainer(SAETrainer):
    """
    Standard SAE training scheme for cross-coding.
    """

    def __init__(
        self,
        dict_class=CrossCoder,
        num_layers=2,
        activation_dim=512,
        dict_size=64 * 512,
        lr=1e-3,
        l1_penalty=1e-1,
        warmup_steps=1000,  # lr warmup period at start of training and after each resample
        resample_steps=None,  # how often to resample neurons
        seed=None,
        device=None,
        layer=None,
        lm_name=None,
        wandb_name="CrossCoderTrainer",
        submodule_name=None,
        compile=False,
        dict_class_kwargs={},
        pretrained_ae=None,
    ):
        super().__init__(seed)

        assert layer is not None and lm_name is not None
        self.layer = layer
        self.lm_name = lm_name
        self.submodule_name = submodule_name
        self.compile = compile
        if seed is not None:
            th.manual_seed(seed)
            th.cuda.manual_seed_all(seed)

        # initialize dictionary
        if pretrained_ae is None:
            self.ae = dict_class(
                activation_dim, dict_size, num_layers=num_layers, **dict_class_kwargs
            )
        else:
            self.ae = pretrained_ae

        if compile:
            self.ae = th.compile(self.ae)
        self.lr = lr
        self.l1_penalty = l1_penalty
        self.warmup_steps = warmup_steps
        self.wandb_name = wandb_name

        if device is None:
            self.device = "cuda" if th.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.ae.to(self.device)

        self.resample_steps = resample_steps

        if self.resample_steps is not None:
            # how many steps since each neuron was last activated?
            self.steps_since_active = th.zeros(self.ae.dict_size, dtype=int).to(
                self.device
            )
        else:
            self.steps_since_active = None

        self.optimizer = th.optim.Adam(self.ae.parameters(), lr=lr)
        if resample_steps is None:

            def warmup_fn(step):
                return min(step / warmup_steps, 1.0)

        else:

            def warmup_fn(step):
                return min((step % resample_steps) / warmup_steps, 1.0)

        self.scheduler = th.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=warmup_fn
        )

    def resample_neurons(self, deads, activations):
        with th.no_grad():
            if deads.sum() == 0:
                return
            self.ae.resample_neurons(deads, activations)
            # reset Adam parameters for dead neurons
            state_dict = self.optimizer.state_dict()["state"]
            ## encoder weight
            state_dict[0]["exp_avg"][:, :, deads] = 0.0
            state_dict[0]["exp_avg_sq"][:, :, deads] = 0.0
            ## encoder bias
            state_dict[1]["exp_avg"][deads] = 0.0
            state_dict[1]["exp_avg_sq"][deads] = 0.0
            ## decoder weight
            state_dict[3]["exp_avg"][:, deads, :] = 0.0
            state_dict[3]["exp_avg_sq"][:, deads, :] = 0.0

    def loss(self, x, logging=False, return_deads=False, **kwargs):
        x_hat, f = self.ae(x, output_features=True)
        l2_loss = th.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        deads = (f <= 1e-8).all(dim=0)
        if self.steps_since_active is not None:
            # update steps_since_active
            self.steps_since_active[deads] += 1
            self.steps_since_active[~deads] = 0

        loss = l2_loss + self.l1_penalty * l1_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "x_hat", "f", "losses"])(
                x,
                x_hat,
                f,
                {
                    "l2_loss": l2_loss.item(),
                    "mse_loss": (x - x_hat).pow(2).sum(dim=-1).mean().item(),
                    "sparsity_loss": l1_loss.item(),
                    "loss": loss.item(),
                    "deads": deads if return_deads else None,
                },
            )

    def update(self, step, activations):
        activations = activations.to(self.device)

        self.optimizer.zero_grad()
        loss = self.loss(activations)
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        if self.resample_steps is not None and step % self.resample_steps == 0:
            self.resample_neurons(
                self.steps_since_active > self.resample_steps / 2, activations
            )

    @property
    def config(self):
        return {
            "dict_class": (
                self.ae.__class__.__name__
                if not self.compile
                else self.ae._orig_mod.__class__.__name__
            ),
            "trainer_class": self.__class__.__name__,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "lr": self.lr,
            "l1_penalty": self.l1_penalty,
            "warmup_steps": self.warmup_steps,
            "resample_steps": self.resample_steps,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
        }
