from collections import namedtuple

import torch
import torch.autograd as autograd
from torch import nn

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


class TrainerJumpRelu(nn.Module, SAETrainer):
    """
    Trains a JumpReLU autoencoder.

    Note does not use learning rate or sparsity scheduling as in the paper.
    """

    def __init__(
        self,
        dict_class=JumpReluAutoEncoder,
        activation_dim=512,
        dict_size=8192,
        steps=30000,
        # XXX: Training decay is not implemented
        seed=None,
        # TODO: What's the default lr use in the paper?
        lr=7e-5,
        bandwidth=0.001,
        sparsity_penalty=0.1,
        device="cpu",
        layer=None,
        lm_name=None,
        wandb_name="JumpRelu",
        submodule_name=None,
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

        self.logging_parameters = []

    def loss(self, x, logging=False, **_):
        f = self.ae.encode(x)
        recon = self.ae.decode(f)

        recon_loss = (x - recon).pow(2).sum(dim=-1).mean()
        l0 = StepFunction.apply(f, self.ae.threshold, self.bandwidth).sum(dim=-1).mean()
        sparsity_loss = self.sparsity_coefficient * l0
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
            "trainer_class": "TrainerJumpRelu",
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
        }
