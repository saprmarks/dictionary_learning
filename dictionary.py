"""
Defines the dictionary classes
"""

from abc import ABC, abstractclassmethod, abstractmethod
import torch as th
import torch.nn as nn
import torch.nn.init as init
from torch.nn.functional import relu
import einops


class Dictionary(ABC, nn.Module):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """

    dict_size: int  # number of features in the dictionary
    activation_dim: int  # dimension of the activation vectors

    @abstractmethod
    def encode(self, x):
        """
        Encode a vector x in the activation space.
        """
        pass

    @abstractmethod
    def decode(self, f):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        pass

    @classmethod
    @abstractmethod
    def from_pretrained(cls, path, device=None, **kwargs) -> "Dictionary":
        """
        Load a pretrained dictionary from a file.
        """
        pass


class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(th.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = th.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x):
        return nn.ReLU()(self.encoder(x - self.bias))

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, ghost_mask=None):
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
            f_ghost = th.exp(f_pre) * ghost_mask.to(f_pre)
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
    def from_pretrained(cls, path, dtype=th.float, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = th.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = cls(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(dtype=dtype, device=device)
        return autoencoder


class IdentityDict(Dictionary, nn.Module):
    """
    An identity dictionary, i.e. the identity function.
    """

    def __init__(self, activation_dim=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = activation_dim

    def encode(self, x):
        return x

    def decode(self, f):
        return f

    def forward(self, x, output_features=False, ghost_mask=None):
        if output_features:
            return x, x
        else:
            return x

    @classmethod
    def from_pretrained(cls, path, dtype=th.float, device=None):
        """
        Load a pretrained dictionary from a file.
        """
        return cls(None)


class GatedAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with separate gating and magnitude networks.
    """

    def __init__(
        self, activation_dim, dict_size, initialization="default", device=None
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.decoder_bias = nn.Parameter(th.empty(activation_dim, device=device))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False, device=device)
        self.r_mag = nn.Parameter(th.empty(dict_size, device=device))
        self.gate_bias = nn.Parameter(th.empty(dict_size, device=device))
        self.mag_bias = nn.Parameter(th.empty(dict_size, device=device))
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False, device=device)
        if initialization == "default":
            self._reset_parameters()
        else:
            initialization(self)

    def _reset_parameters(self):
        """
        Default method for initializing GatedSAE weights.
        """
        # biases are initialized to zero
        init.zeros_(self.decoder_bias)
        init.zeros_(self.r_mag)
        init.zeros_(self.gate_bias)
        init.zeros_(self.mag_bias)

        # decoder weights are initialized to random unit vectors
        dec_weight = th.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x, return_gate=False):
        """
        Returns features, gate value (pre-Heavyside)
        """
        x_enc = self.encoder(x - self.decoder_bias)

        # gating network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).to(self.encoder.weight.dtype)

        # magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = nn.ReLU()(pi_mag)

        f = f_gate * f_mag

        # W_dec norm is not kept constant, as per Anthropic's April 2024 Update
        # Normalizing after encode, and renormalizing before decode to enable comparability
        f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if return_gate:
            return f, nn.ReLU()(pi_gate)

        return f

    def decode(self, f):
        # W_dec norm is not kept constant, as per Anthropic's April 2024 Update
        # Normalizing after encode, and renormalizing before decode to enable comparability
        f = f / self.decoder.weight.norm(dim=0, keepdim=True)
        return self.decoder(f) + self.decoder_bias

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)

        f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if output_features:
            return x_hat, f
        else:
            return x_hat

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = th.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = GatedAutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class JumpReluAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with jump ReLUs.
    """

    def __init__(self, activation_dim, dict_size, device="cpu"):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.W_enc = nn.Parameter(th.empty(activation_dim, dict_size, device=device))
        self.b_enc = nn.Parameter(th.zeros(dict_size, device=device))
        self.W_dec = nn.Parameter(th.empty(dict_size, activation_dim, device=device))
        self.b_dec = nn.Parameter(th.zeros(activation_dim, device=device))
        self.threshold = nn.Parameter(th.zeros(dict_size, device=device))

        self.apply_b_dec_to_input = False

        # rows of decoder weight matrix are initialized to unit vectors
        self.W_enc.data = th.randn_like(self.W_enc)
        self.W_enc.data = self.W_enc / self.W_enc.norm(dim=0, keepdim=True)
        self.W_dec.data = self.W_enc.data.clone().T

    def encode(self, x, output_pre_jump=False):
        if self.apply_b_dec_to_input:
            x = x - self.b_dec
        pre_jump = x @ self.W_enc + self.b_enc

        f = nn.ReLU()(pre_jump * (pre_jump > self.threshold))
        f = f * self.W_dec.norm(dim=1)

        if output_pre_jump:
            return f, pre_jump
        else:
            return f

    def decode(self, f):
        f = f / self.W_dec.norm(dim=1)
        return f @ self.W_dec + self.b_dec

    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features (and their pre-jump version) as well as the decoded x
        """
        f = self.encode(x)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        else:
            return x_hat

    @classmethod
    def from_pretrained(
        cls,
        path: str | None = None,
        load_from_sae_lens: bool = False,
        dtype: th.dtype = th.float32,
        device: th.device | None = None,
        **kwargs,
    ):
        """
        Load a pretrained autoencoder from a file.
        If sae_lens=True, then pass **kwargs to sae_lens's
        loading function.
        """
        if not load_from_sae_lens:
            state_dict = th.load(path)
            dict_size, activation_dim = state_dict["W_enc"].shape
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size)
            autoencoder.load_state_dict(state_dict)
        else:
            from sae_lens import SAE

            sae, cfg_dict, _ = SAE.from_pretrained(**kwargs)
            assert (
                cfg_dict["finetuning_scaling_factor"] == False
            ), "Finetuning scaling factor not supported"
            dict_size, activation_dim = cfg_dict["d_sae"], cfg_dict["d_in"]
            autoencoder = JumpReluAutoEncoder(activation_dim, dict_size, device=device)
            autoencoder.load_state_dict(sae.state_dict())
            autoencoder.apply_b_dec_to_input = cfg_dict["apply_b_dec_to_input"]

        if device is not None:
            device = autoencoder.W_enc.device
        return autoencoder.to(dtype=dtype, device=device)


# TODO merge this with AutoEncoder
class AutoEncoderNew(Dictionary, nn.Module):
    """
    The autoencoder architecture and initialization used in https://transformer-circuits.pub/2024/april-update/index.html#training-saes
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)

        # initialize encoder and decoder weights
        w = th.randn(activation_dim, dict_size)
        ## normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        ## set encoder and decoder weights
        self.encoder.weight = nn.Parameter(w.clone().T)
        self.decoder.weight = nn.Parameter(w.clone())

        # initialize biases to zeros
        init.zeros_(self.encoder.bias)
        init.zeros_(self.decoder.bias)

    def encode(self, x):
        return nn.ReLU()(self.encoder(x))

    def decode(self, f):
        return self.decoder(f)

    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        """
        if not output_features:
            return self.decode(self.encode(x))
        else:  # TODO rewrite so that x_hat depends on f
            f = self.encode(x)
            x_hat = self.decode(f)
            # multiply f by decoder column norms
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)
            return x_hat, f

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = th.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = AutoEncoderNew(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class CrossCoderEncoder(nn.Module):
    """
    A cross-coder encoder
    """

    def __init__(
        self,
        activation_dim,
        dict_size,
        num_layers,
        same_init_for_all_layers: bool = False,
        norm_init_scale: float | None = None,
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_layers = num_layers
        if same_init_for_all_layers:
            weight = init.kaiming_uniform_(th.empty(activation_dim, dict_size))
            weight = weight.repeat(num_layers, 1, 1)
        else:
            weight = init.kaiming_uniform_(
                th.empty(num_layers, activation_dim, dict_size)
            )
        if norm_init_scale is not None:
            weight = weight / weight.norm(dim=1, keepdim=True) * norm_init_scale
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(th.zeros(dict_size))

    def forward(self, x: th.Tensor) -> th.Tensor:  # (batch_size, activation_dim)
        """
        Convert activations to features for each layer

        Args:
            x: (batch_size, n_layers, activation_dim)
        Returns:
            f: (batch_size, dict_size)
        """
        f = th.einsum("bld, ldD -> blD", x, self.weight).sum(dim=1)
        return relu(f + self.bias)


class CrossCoderDecoder(nn.Module):
    """
    A cross-coder decoder
    """

    def __init__(
        self,
        activation_dim,
        dict_size,
        num_layers,
        same_init_for_all_layers: bool = False,
        norm_init_scale: float | None = None,
        init_with_weight: th.Tensor | None = None,
    ):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_layers = num_layers
        self.bias = nn.Parameter(th.zeros(num_layers, activation_dim))
        if init_with_weight is not None:
            self.weight = nn.Parameter(init_with_weight)
        else:
            if same_init_for_all_layers:
                weight = init.kaiming_uniform_(th.empty(dict_size, activation_dim))
                weight = weight.repeat(num_layers, 1, 1)
            else:
                weight = init.kaiming_uniform_(
                    th.empty(num_layers, dict_size, activation_dim)
                )
            if norm_init_scale is not None:
                raise NotImplementedError(
                    "Normalized initialization not implemented for crosscoder"
                )
            self.weight = nn.Parameter(weight)

    def forward(
        self, f: th.Tensor
    ) -> th.Tensor:  # (batch_size, n_layers, activation_dim)
        # f: (batch_size, n_layers, dict_size)
        """
        Convert features to activations for each layer

        Args:
            f: (batch_size, dict_size)
        Returns:
            x: (batch_size, n_layers, activation_dim)
        """
        return th.einsum("bD, lDd -> bld", f, self.weight) + self.bias


class CrossCoder(Dictionary, nn.Module):
    """
    A cross-coder using the AutoEncoderNew architecture for two models.

    encoder: shape (num_layers, activation_dim, dict_size)
    decoder: shape (num_layers, dict_size, activation_dim)
    """

    def __init__(
        self,
        activation_dim,
        dict_size,
        num_layers,
        same_init_for_all_layers=False,
        norm_init_scale: float | None = None,  # neel's default: 0.005
        init_with_transpose=True,
    ):
        """
        Args:
            same_init_for_all_layers: if True, initialize all layers with the same vector
            norm_init_scale: if not None, initialize the weights with a norm of this value
            init_with_transpose: if True, initialize the decoder weights with the transpose of the encoder weights
        """
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.num_layers = num_layers

        self.encoder = CrossCoderEncoder(
            activation_dim,
            dict_size,
            num_layers,
            same_init_for_all_layers=same_init_for_all_layers,
            norm_init_scale=norm_init_scale,
        )
        if init_with_transpose:
            decoder_weight = einops.rearrange(
                self.encoder.weight.data.clone(),
                "num_layers activation_dim dict_size -> num_layers dict_size activation_dim",
            )
        else:
            decoder_weight = None
        self.decoder = CrossCoderDecoder(
            activation_dim,
            dict_size,
            num_layers,
            same_init_for_all_layers=same_init_for_all_layers,
            init_with_weight=decoder_weight,
            norm_init_scale=norm_init_scale,
        )

    def encode(self, x: th.Tensor) -> th.Tensor:  # (batch_size, n_layers, dict_size)
        # x: (batch_size, n_layers, activation_dim)
        return self.encoder(x)

    def decode(
        self, f: th.Tensor
    ) -> th.Tensor:  # (batch_size, n_layers, activation_dim)
        # f: (batch_size, n_layers, dict_size)
        return self.decoder(f)

    def forward(self, x: th.Tensor, output_features=False):
        """
        Forward pass of the cross-coder.
        x : activations to be encoded and decoded
        output_features : if True, return the encoded features as well as the decoded x
        """
        f = self.encode(x)
        x_hat = self.decode(f)

        if output_features:
            # Scale features by decoder column norms
            f_scaled = f * self.decoder.weight.norm(dim=2).sum(
                dim=0, keepdim=True
            )  # Also sum across layers for the loss
            return x_hat, f_scaled
        else:
            return x_hat

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        dtype: th.dtype = th.float32,
        device: th.device | None = None,
    ):
        """
        Load a pretrained cross-coder from a file.
        """
        state_dict = th.load(path, map_location="cpu", weights_only=True)
        num_layers, activation_dim, dict_size = state_dict["encoder.weight"].shape
        cross_coder = cls(activation_dim, dict_size, num_layers)
        cross_coder.load_state_dict(state_dict)

        if device is not None:
            cross_coder = cross_coder.to(device)
        return cross_coder.to(dtype=dtype)
