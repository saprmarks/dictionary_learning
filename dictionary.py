"""
Defines the dictionary classes
"""

from abc import ABC, abstractmethod
import torch as t
import torch.nn as nn

from .grad_pursuit import grad_pursuit

class Dictionary(ABC):
    """
    A dictionary consists of a collection of vectors, an encoder, and a decoder.
    """
    dict_size : int # number of features in the dictionary
    activation_dim : int # dimension of the activation vectors

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
        if ghost_mask is None: # normal mode
            f = self.encode(x)
            x_hat = self.decode(f)
            if output_features:
                return x_hat, f
            else:
                return x_hat
        
        else: # ghost mode
            f_pre = self.encoder(x - self.bias)
            f_ghost = t.exp(f_pre) * ghost_mask.to(f_pre)
            f = nn.ReLU()(f_pre)

            x_ghost = self.decoder(f_ghost) # note that this only applies the decoder weight matrix, no bias
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
        dict_size, activation_dim = state_dict['encoder.weight'].shape
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
    def __init__(self, activation_dim, dict_size):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.decoder_bias = nn.Parameter(t.zeros(activation_dim))
        self.encoder = nn.Linear(activation_dim, dict_size, bias=False)
        self.r_mag = nn.Parameter(t.zeros(dict_size))
        self.gate_bias = nn.Parameter(t.zeros(dict_size))
        self.mag_bias = nn.Parameter(t.zeros(dict_size))

        # rows of decoder weight matrix are unit vectors
        self.decoder = nn.Linear(dict_size, activation_dim, bias=False)
        dec_weight = t.randn_like(self.decoder.weight)
        dec_weight = dec_weight / dec_weight.norm(dim=0, keepdim=True)
        self.decoder.weight = nn.Parameter(dec_weight)

    def encode(self, x, gate_only=False):
        """
        If gate_only is True, instead return ReLU(gating preactivations)
        """
        if not gate_only:
            # gating network
            pi_gate = self.encoder(x - self.decoder_bias) + self.gate_bias
            f_gate = (pi_gate > 0).float()

            # magnitude network
            pi_mag = self.r_mag.exp() * self.encoder(x - self.decoder_bias) + self.mag_bias
            f_mag = nn.ReLU()(pi_mag)

            return f_gate * f_mag
        else:
            # only gating network
            pi_gate = self.encoder(x - self.decoder_bias) + self.gate_bias
            return nn.ReLU()(pi_gate)

    def decode(self, f):
        return self.decoder(f) + self.decoder_bias
    
    def forward(self, x, output_features=False, gate_only=False):
        f = self.encode(x, gate_only=gate_only)
        x_hat = self.decode(f)
        if output_features:
            return x_hat, f
        else:
            return x_hat

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
        else: # TODO rewrite so that x_hat depends on f
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
        dict_size, activation_dim = state_dict['encoder.weight'].shape
        autoencoder = AutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder


class GradPursuitAutoEncoder(Dictionary, nn.Module):
    """
    Uses gradient pursuit as encoder.
    """
    def __init__(self, activation_dim, dict_size, target_l0=20, device='cpu'):
        super().__init__()
        self.activation_dim = activation_dim
        self.dict_size = dict_size
        self.decoder = nn.Linear(dict_size, activation_dim, bias=True)

        # initialize encoder and decoder weights
        w = t.randn(activation_dim, dict_size)
        ## normalize columns of w
        w = w / w.norm(dim=0, keepdim=True) * 0.1
        ## set encoder and decoder weights
        self.decoder.weight = nn.Parameter(w.clone())
        self.target_l0 = target_l0
        self.device = device

    def encode(self, x):
        return grad_pursuit(x, self.decoder.weight, target_l0=self.target_l0, device=self.device)
    
    def decode(self, f):
        return self.decoder(f)
    
    def forward(self, x, output_features=False):
        """
        Forward pass of an autoencoder.
        x : activations to be autoencoded
        """
        if not output_features:
            return self.decode(self.encode(x))
        else: # TODO rewrite so that x_hat depends on f
            f = self.encode(x)
            x_hat = self.decode(f)
            # multiply f by decoder column norms
            f = f * self.decoder.weight.norm(dim=0, keepdim=True)
            return x_hat, f
            
    def from_pretrained(path, device=None):
        """
        Load a pretrained autoencoder from a file.
        """
        assert False # TODO: need to make sure we load this correctly
        state_dict = t.load(path)
        dict_size, activation_dim = state_dict['encoder.weight'].shape
        autoencoder = AutoEncoder(activation_dim, dict_size)
        autoencoder.load_state_dict(state_dict)
        if device is not None:
            autoencoder.to(device)
        return autoencoder