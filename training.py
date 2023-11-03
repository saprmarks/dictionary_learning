"""
Training dictionaries
"""

import torch as t
from .dictionary import AutoEncoder
from einops import einsum
from .buffer import ActivationBuffer

class ConstrainedAdam(t.optim.Adam):
    """
    A variant of Adam where some of the parameters are constrained to have unit norm.
    """
    def __init__(self, params, constrained_params, lr):
        super().__init__(params, lr=lr)
        self.constrained_params = list(constrained_params)
    
    def step(self, closure=None):
        with t.no_grad():
            for p in self.constrained_params:
                # project away the parallel component of the gradient
                p.grad -= einsum(p.grad, p, '... i, ... i -> ...').unsqueeze(-1) * p / p.norm(dim=-1, keepdim=True)**2
        super().step(closure=closure)

        # project the constrained parameters back onto the unit sphere
        with t.no_grad():
            for p in self.constrained_params:
                p /= p.norm(dim=-1, keepdim=True)

def entropy(p):
    eps = 1e-8
    # Calculate the sum along the last dimension (i.e., sum of each vector in the batch)
    p_sum = p.sum(dim=-1, keepdim=True)
    
    # Avoid in-place operations that can interfere with autograd
    p_normed = p / (p_sum + eps)  # Add eps to prevent division by zero
    
    # Compute the log safely, adding eps inside the log to prevent log(0)
    p_log = t.log(p_normed + eps)  # Add eps to prevent log(0)

    # Compute the entropy, this will give zero for elements where p_normed is zero
    ent = -(p_normed * p_log)
    
    # Zero out the entropy where the sum of p is zero (i.e., for all-zero vectors)
    ent = t.where(p_sum > 0, ent, t.zeros_like(ent))

    # Sum the entropy across the features and then take the mean across the batch
    return ent.sum(dim=-1).mean()


def sae_loss(activations, ae, sparsity_penalty, use_entropy=False, separate=False):
    """
    Compute the loss of an autoencoder on some activations
    """
    f = ae.encode(activations)
    x_hat = ae.decode(f)
    mse_loss = t.nn.MSELoss()(
        activations.reshape(-1, activations.shape[-1]), 
        x_hat.reshape(-1, x_hat.shape[-1])
    )
    if use_entropy:
        sparsity_loss = entropy(f)
    else:
        sparsity_loss = f.norm(p=1, dim=-1).mean()
    if separate:
        return mse_loss, sparsity_loss
    else:
        return mse_loss + sparsity_penalty * sparsity_loss
    
def reconstruction_loss(
        tokens, # a tokenized batch
        model,
        submodule,
        ae, # an AutoEncoder
        pct=False # return pct recovered; if False, return losses
):
    """
    Compute the reconstruction loss of model on a batch of data
    This is the model's loss if the component output is replaced with the reconstruction by the autoencoder.
    """
    
    # unmodified logits
    with model.forward(tokens):
        logits_original = model.embed_out.output.save()
    
    # logits when replacing component output with reconstruction by autoencoder
    with model.forward(tokens):
        submodule.output = ae(submodule.output)
        logits_reconstructed = model.embed_out.output.save()
    
    # logits when zero ablating component
    with model.forward(tokens):
        submodule.output = t.zeros_like(submodule.output)
        logits_zero = model.embed_out.output.save()
    
    losses = []
    for logits in [logits_original, logits_reconstructed, logits_zero]:
        loss = t.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)(
            logits.value[:,:-1,:].reshape(-1, logits.value.shape[-1]),
            tokens[:,1:].reshape(-1)
        ).item()
        losses.append(loss)
    
    if pct:
        return (losses[1] - losses[2]) / (losses[0] - losses[2])
    else:
        return tuple(losses)

def trainSAE(
        activations,
        activation_dim, 
        dictionary_size,
        lr, 
        sparsity_penalty,
        entropy=False,
        steps=None,
        log_steps=100,
        device='cpu'):
    """
    Train a sparse autoencoder
    """
    # initialize the dictionary
    ae = AutoEncoder(activation_dim, dictionary_size).to(device)

    # train the dictionary
    optimizer = ConstrainedAdam(ae.parameters(), ae.decoder.parameters(), lr=lr)
    #optimizer = t.optim.Adam(ae.parameters(), lr=lr)

    for step, acts in enumerate(activations):
        if steps is not None and step >= steps:
            break
        acts = acts.to(device)
        optimizer.zero_grad()
        loss = sae_loss(acts, ae, sparsity_penalty, entropy, separate=False)
        loss.backward()
        optimizer.step()
        if log_steps is not None and step % log_steps == 0:
            with t.no_grad():
                print(f"step {step} autoencoder loss: {loss.item()}")
                dict_acts = ae.encode(acts)
                print(f"step {step} % inactive: {(dict_acts == 0).all(dim=0).sum() / dict_acts.shape[-1]}")
                if isinstance(activations, ActivationBuffer):
                    tokens = activations.tokenized_batch().input_ids
                    loss_orig, loss_reconst, loss_zero = reconstruction_loss(tokens, activations.model, activations.submodule, ae)
                    print(f"step {step} reconstruction loss: {loss_orig}, {loss_reconst}, {loss_zero}")

    return ae

