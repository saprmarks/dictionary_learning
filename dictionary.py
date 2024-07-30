#%%
"""
Defines the dictionary classes
"""

from abc import ABC, abstractmethod
import torch as t
import torch.nn as nn
import torch.nn.init as init


class Dictionary(ABC):
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


class AutoEncoder(Dictionary, nn.Module):
    """
    A one-layer autoencoder.
    """

    def __init__(self, activation_dim, dict_size):
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

    def from_pretrained(path, device=None):
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


class GatedAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with separate gating and magnitude networks.
    """

    def __init__(self, activation_dim, dict_size, initialization="default", device=None):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.decoder_bias = nn.Parameter(t.empty(activation_dim, device=device))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False, device=device)
        self.r_mag = nn.Parameter(t.empty(dict_size, device=device))
        self.gate_bias = nn.Parameter(t.empty(dict_size, device=device))
        self.mag_bias = nn.Parameter(t.empty(dict_size, device=device))
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
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x, return_gate=False):
        """
        Returns features, gate value (pre-Heavyside)
        """
        x_enc = self.encoder(x - self.decoder_bias)

        # gating network
        pi_gate = x_enc + self.gate_bias
        f_gate = (pi_gate > 0).float()

        # magnitude network
        pi_mag = self.r_mag.exp() * x_enc + self.mag_bias
        f_mag = nn.ReLU()(pi_mag)

        f = f_gate * f_mag

        if return_gate:
            return f, nn.ReLU()(pi_gate)

        return f

    def decode(self, f):
        return self.decoder(f) + self.decoder_bias

    def forward(self, x, output_features=False):
        f = self.encode(x)
        x_hat = self.decode(f)

        # TODO: modify so that x_hat depends on f
        f = f * self.decoder.weight.norm(dim=0, keepdim=True)

        if output_features:
            return x_hat, f
        else:
            return x_hat

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = GatedAutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class JumpAutoEncoder(Dictionary, nn.Module):
    """
    An autoencoder with jump ReLUs. Replacement for GatedAutoEncoder.
    """

    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)

        # jump values added to activated features
        self.jump = nn.Parameter(t.zeros(dict_size))

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x, output_pre_jump=False):
        pre_jump = nn.ReLU()(self.encoder(x - self.bias))
        f = pre_jump + self.jump * (pre_jump > 0)
        if output_pre_jump:
            return f, pre_jump
        else:
            return f

    def decode(self, f):
        return self.decoder(f) + self.bias

    def forward(self, x, output_features=False, output_pre_jump=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        output_features : if True, return the encoded features (and their pre-jump version) as well as the decoded x
        """
        f, pre_jump = self.encode(x, output_pre_jump=True)
        x_hat = self.decode(f)
        if output_pre_jump:
            return x_hat, f, pre_jump
        elif output_features:
            return x_hat, f
        else:
            return x_hat

    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = JumpAutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


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
        w = t.randn(activation_dim, dict_size)
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
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = AutoEncoderNew(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class AutoEncoderTopKnotriton(Dictionary, nn.Module):
    """
    Adaptation of The top-k autoencoder without architecture and initialization used in https://arxiv.org/abs/2406.04093
    - without Triton integration
    - apply TopK in encoder step
    """

    def __init__(self, activation_dim, dict_size, k):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.k = k

        # self.encoder = nn.Linear(activation_dim, dict_size)
        # self.encoder.bias.data.zero_()
        # print(self.encoder.weight.data.shape)

        # self.decoder = nn.Linear(self.encoder.weight.data.clone())
        # self.set_decoder_norm_to_unit_norm()
        # print(self.decoder.data.shape)

        self.encoder = nn.Linear(activation_dim, dict_size, bias=True)

        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        self.decoder.weight = nn.Parameter(self.encoder.weight.data.t().clone())
        self.set_decoder_norm_to_unit_norm()

        self.b_dec = self.b_dec = nn.Parameter(t.zeros(activation_dim))

    def encode(self, x):
        post_nonlinearity = nn.functional.relu(self.encoder(x - self.b_dec))
        post_topk_acts, post_topk_idxs = post_nonlinearity.topk(self.k, sorted=False, keepdim=True, dim=-1)
        post_topk_dense = t.zeros(post_nonlinearity.shape)
        post_topk_dense.scatter_(-1, post_topk_idxs, post_topk_acts)
        return post_topk_dense
    
    def decode(self, post_topK):
        return self.decoder(post_topK) + self.b_dec

    def encode_sparse(self, x):
        post_nonlinearity = nn.functional.relu(self.encoder(x - self.b_dec))
        latent_shape = post_nonlinearity.shape
        flattened_latents = post_nonlinearity.view(-1)
        post_topK_acts, post_topK_idxs = flattened_latents.topk(self.k, sorted=False)
        return post_topK_acts, post_topK_idxs, latent_shape

    def decode_sparse(self, post_topK, post_topK_idxs, latent_shape):
        """
        Decode a dictionary vector f (i.e. a linear combination of dictionary elements)
        """
        d = t.zeros(latent_shape).flatten()
        d[post_topK_idxs] = post_topK
        d = d.view(latent_shape)
        print(f'self.decoder.shape: {self.decoder.shape}')
        print(f'd.shape: {d.shape}')
        d = d @ self.decoder
        return d + self.b_dec

    def forward(self, x, output_features=False):
        post_topK_acts, post_topK_idxs, latent_shape = self.encode_sparse(x)
        x_hat = self.decode_sparse(post_topK_acts, post_topK_idxs, latent_shape)
        if not output_features:
            return x_hat
        else:
            return x_hat, post_topK_acts, post_topK_idxs, latent_shape

    # @t.no_grad()
    # def set_decoder_norm_to_unit_norm(self):
    #     eps = t.finfo(self.decoder.dtype).eps
    #     norm = t.norm(self.decoder.data, dim=1, keepdim=True)
    #     self.decoder.data /= norm + eps

    @t.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        eps = t.finfo(self.decoder.weight.dtype).eps
        norm = t.norm(self.decoder.weight.data, dim=0, keepdim=True)
        self.decoder.weight.data /= norm + eps

    def from_pretrained(path, k, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = AutoEncoderTopKnotriton(activation_dim, dict_size, k)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder
    
    def from_oai_format(path, k, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict["encoder.weight"].shape
        autoencoder = AutoEncoderTopKnotriton(activation_dim, dict_size, k)
        autoencoder.encoder = state_dict["encoder"]
        autoencoder.decoder = state_dict["decoder"]
        if device is not None:
            autoencoder.to(device)
        return autoencoder


# %%
import torch

# %%
B, S = 3, 64

activation_dim = 512
k = 128
dict_size = 1024
device = 'cpu'
ae = AutoEncoderTopKnotriton(activation_dim, dict_size, k).to(device)

random_input = torch.randn(B, S, activation_dim).to(device)

direct_output = ae(random_input)
latents = ae.encode(random_input)
decoded_output = ae.decode(*latents)
