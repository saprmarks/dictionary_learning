"""
Training dictionaries
"""

import torch as t
from dictionary import AutoEncoder

def sae_loss(activations, ae, l1_penalty):
    """
    Compute the reconstruction loss of an autoencoder on some activations
    """
    f = ae.encode(activations)
    x_hat = ae.decode(f)
    return t.nn.MSELoss()(x_hat, activations) + l1_penalty * t.norm(f, 1)


def trainSAE(
        buffer, 
        dictionary_size,
        steps,
        lr, 
        l1_penalty,
        device):
    """
    Train a sparse autoencoder
    """
    # initialize the dictionary
    activation_dim = buffer.submodule.out_features # this should maybe be made more general
    ae = AutoEncoder(activation_dim, dictionary_size).to(device)

    # train the dictionary
    optimizer = t.optim.Adam(ae.parameters(), lr=lr)

    for _ in range(steps):
        acts = buffer.get_batch()
        acts = acts.to(device)
        optimizer.zero_grad()
        loss = sae_loss(acts, ae, l1_penalty)
        loss.backward()
        optimizer.step()
        if _ % 100 == 0:
            print(loss.item())

