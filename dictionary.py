"""
Defines the dictionary classes
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch as t
import torch.nn as nn


class Dictionary(ABC):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """

    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    @abstractmethod
    def encode(self, x: t.Tensor) -> t.Tensor:
        """
        Encode a vector x in the activation space.
        """
        pass

    @abstractmethod
    def decode(self, f: t.Tensor) -> t.Tensor:
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass


class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    """

    def __init__(self, activation_dim: int, dict_size: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x: t.Tensor) -> t.Tensor:
        return nn.ReLU()(self.encoder(x - self.bias))

    def decode(self, f: t.Tensor) -> t.Tensor:
        return self.decoder(f) + self.bias

    def forward(
        self, x: t.Tensor, output_features: bool = False, ghost_mask: Optional[t.Tensor] = None
    ):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features as well as the decoded x
        ghost_mask : if not None, run this autoencoder in "ghost mode" where features are masked
        """
        if ghost_mask is None:  # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, f
            else:
                return x_hat

        else:  # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(
                f_ghost
            )  # note that this only applies the decoder weight matrix, no bias
            x_hat = self.decode(f)
            if output_features:
                return x_hat, x_ghost, f
            else:
                return x_hat, x_ghost

    @classmethod
    def from_pretrained(cls, path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = AutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class IdentityDict(Dictionary, nn.Module):
    """
    An identity dictionary, i.e. the identity function.
    """

    def __init__(self, activation_dim: Optional[int] = None):
        super().__init__()

    def encode(self, x: t.Tensor) -> t.Tensor:
        return x

    def decode(self, f: t.Tensor) -> t.Tensor:
        return f

    def forward(
        self, x: t.Tensor, output_features: bool = False, ghost_mask: Optional[t.Tensor] = None
    ):
        if output_features:
            return x, x
        else:
            return x
