from collections import namedtuple

import torch
import torch.autograd as autograd
from torch import nn
from typing import Optional

from ..dictionary import Dictionary, JumpReluAutoEncoder
from .trainer import SAETrainer


class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class JumpReluTrainer(nn.Module, SAETrainer):
    """
    Trains a JumpReLU autoencoder.

    Note does not use learning rate or sparsity scheduling as in the paper.
    """
    def __init__(
        self,
        steps: int, # total number of steps to train for
        activation_dim: int,
        dict_size: int,
        layer: int,
        lm_name: str,
        dict_class=JumpReluAutoEncoder,
        # XXX: Training decay is not implemented
        seed: Optional[int] = None,
        # TODO: What's the default lr use in the paper?
        lr: float = 7e-5,
        bandwidth: float = 0.001,
        sparsity_penalty: float = 1.0,
        warmup_steps:int=1000, # lr warmup period at start of training and after each resample
        sparsity_warmup_steps:Optional[int]=2000, # sparsity warmup period at start of training
        decay_start:Optional[int]=None, # decay learning rate after this many steps
        target_l0: float = 20.0,
        device: str = "cpu",
        wandb_name: str = "JumpRelu",
        submodule_name: Optional[str] = None,
    ):
        super().__init__()

        # TODO: Should just be args, and this should be commonised
        assert layer is not None, "Layer must be specified"
        assert lm_name is not None, "Language model name must be specified"
        self.lm_name = lm_name
        self.layer = layer
        self.submodule_name = submodule_name
        self.device = device
        self.steps = steps
        self.lr = lr
        self.seed = seed

        self.bandwidth = bandwidth
        self.sparsity_coefficient = sparsity_penalty
        self.warmup_steps = warmup_steps
        self.sparsity_warmup_steps = sparsity_warmup_steps
        self.decay_start = decay_start
        self.target_l0 = target_l0

        # TODO: Better auto-naming (e.g. in BatchTopK package)
        self.wandb_name = wandb_name

        # TODO: Why not just pass in the initialised autoencoder instead?
        self.ae = dict_class(
            activation_dim=activation_dim,
            dict_size=dict_size,
            device=device,
        ).to(self.device)

        # Parameters from the paper
        self.optimizer = torch.optim.Adam(
            self.ae.parameters(), lr=lr, betas=(0.0, 0.999), eps=1e-8
        )

        if decay_start is not None:
            assert 0 <= decay_start < steps, "decay_start must be >= 0 and < steps."
            assert decay_start > warmup_steps, "decay_start must be > warmup_steps."
            if sparsity_warmup_steps is not None:
                assert decay_start > sparsity_warmup_steps, "decay_start must be > sparsity_warmup_steps."

        assert 0 <= warmup_steps < steps, "warmup_steps must be >= 0 and < steps."

        if sparsity_warmup_steps is not None:
            assert 0 <= sparsity_warmup_steps < steps, "sparsity_warmup_steps must be >= 0 and < steps."

        def warmup_fn(step):
            if step < warmup_steps:
                return step / warmup_steps
            
            if decay_start is not None and step >= decay_start:
                return (steps - step) / (steps - decay_start)

            return 1.0
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=warmup_fn)

        self.logging_parameters = []

    def loss(self, x: torch.Tensor, step: int, logging=False, **_):

        if self.sparsity_warmup_steps is not None:
            sparsity_scale = min(step / self.sparsity_warmup_steps, 1.0)
        else:
            sparsity_scale = 1.0

        f = self.ae.encode(x)
        recon = self.ae.decode(f)

        recon_loss = (x - recon).pow(2).sum(dim=-1).mean()
        l0 = StepFunction.apply(f, self.ae.threshold, self.bandwidth).sum(dim=-1).mean()

        sparsity_loss = self.sparsity_coefficient * ((l0 / self.target_l0) - 1).pow(2) * sparsity_scale
        loss = recon_loss + sparsity_loss

        if not logging:
            return loss
        else:
            return namedtuple("LossLog", ["x", "recon", "f", "losses"])(
                x,
                recon,
                f,
                {
                    "l2_loss": recon_loss.item(),
                    "loss": loss.item(),
                },
            )

    def update(self, step, x):
        x = x.to(self.device)
        loss = self.loss(x, step=step)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.ae.parameters(), 1.0)

        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss.item()

    @property
    def config(self):
        return {
            "trainer_class": "JumpReluTrainer",
            "dict_class": "JumpReluAutoEncoder",
            "lr": self.lr,
            "steps": self.steps,
            "seed": self.seed,
            "activation_dim": self.ae.activation_dim,
            "dict_size": self.ae.dict_size,
            "device": self.device,
            "layer": self.layer,
            "lm_name": self.lm_name,
            "wandb_name": self.wandb_name,
            "submodule_name": self.submodule_name,
            "bandwidth": self.bandwidth,
            "sparsity_penalty": self.sparsity_coefficient,
            "sparsity_warmup_steps": self.sparsity_warmup_steps,
            "target_l0": self.target_l0,
        }
