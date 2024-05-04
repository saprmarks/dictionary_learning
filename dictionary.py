"""
Defines the dictionary classes
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

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
    def from_pretrained(cls, path: str, device: Optional[t.device] = None):
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


class GatedAutoEncoder(Dictionary, nn.Module):
    """An autoencoder with a gating mechanism as given in Improving Dictionary Learning with
    Gated Sparse Autoencoders (Rajamanoharan et al 2024)
    """

    def __init__(self, activation_dim: int, dict_size: int):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size

        # Using tied bias term for encoder and decoder
        self.neuron_bias = nn.Parameter(t.zeros(activation_dim))

        self.encoder = nn.Linear(activation_dim, dict_size, bias=False)

        self.feature_magnitude_bias = nn.Parameter(t.zeros(dict_size))
        self.r_magnitude = nn.Parameter(t.zeros(dict_size))

        self.gating_bias = nn.Parameter(t.zeros(dict_size))

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)

        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)
        self.act_fn = nn.ReLU()

    def encode(self, x: t.Tensor) -> tuple[t.Tensor, t.Tensor, t.IntTensor, t.Tensor]:
        # Apply pre-encoder bias
        x_centered = x - self.neuron_bias
        naive_features = self.encoder(x_centered)

        # Magnitudes encoder path (estimates active features’ magnitudes)
        magnitude_scale = t.exp(self.r_magnitude)
        feature_magnitudes = self.act_fn(
            magnitude_scale * naive_features + self.feature_magnitude_bias
        )  # scale and shift

        # Gating encoder path (estimates which features are active)
        active_features_pre_binarisation = naive_features + self.gating_bias
        active_features: t.IntTensor = active_features_pre_binarisation > 0

        # Element-wise multiplication of active features and their magnitudes
        features = active_features * feature_magnitudes

        return features, active_features_pre_binarisation, active_features, feature_magnitudes

    def decode(self, features: t.Tensor) -> t.Tensor:
        features = self.decoder(features) + self.neuron_bias
        return features

    def forward(
        self,
        x: t.Tensor,
        output_features: bool = False,
        output_intermediate_activations: bool = False,
    ):
        """
        Forward pass of the Gated Sparse Autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features, active_features
            and feature_magnitude tensors as well as the decoded x
        """
        features, active_features_pre_binarisation, active_features, feature_magnitudes = (
            self.encode(x)
        )
        x_hat = self.decode(features)
        if output_intermediate_activations:
            return (
                x_hat,
                features,
                active_features_pre_binarisation,
                active_features,
                feature_magnitudes,
            )
        elif output_features:
            return x_hat, features
        else:
            return x_hat

    @classmethod
    def from_pretrained(cls, path: str, device: Optional[t.device] = None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["active_features_encoder.weight"].shape

        autoencoder = GatedAutoEncoder(activation_dim, dict_size)
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


AbstractAutoEncoder = Union[AutoEncoder, GatedAutoEncoder, IdentityDict]
