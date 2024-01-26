"""
Training dictionaries
"""

import torch as t
from .dictionary import AutoEncoder
from .buffer import ActivationBuffer
import os
from tqdm import tqdm

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
                normed_p = p / p.norm(dim=0, keepdim=True)
                # project away the parallel component of the gradient
                p.grad -= (p.grad * normed_p).sum(dim=0, keepdim=True) * normed_p
        super().step(closure=closure)
        with t.no_grad():
            for p in self.constrained_params:
                # renormalize the constrained parameters
                p /= p.norm(dim=0, keepdim=True)

def entropy(p, eps=1e-8):
    p_sum = p.sum(dim=-1, keepdim=True)
    # epsilons for numerical stability    
    p_normed = p / (p_sum + eps)    
    p_log = t.log(p_normed + eps)
    ent = -(p_normed * p_log)
    
    # Zero out the entropy where p_sum is zero
    ent = t.where(p_sum > 0, ent, t.zeros_like(ent))

    return ent.sum(dim=-1).mean()


def sae_loss(activations, ae, sparsity_penalty, use_entropy=False, separate=False, num_samples_since_activated=None, ghost_threshold=None):
    """
    Compute the loss of an autoencoder on some activations
    If separate is True, return the MSE loss, the sparsity loss, and the ghost loss separately
    If num_samples_since_activated is not None, use it to do ghost grads
    If ghost_threshold is not None, use it to do ghost grads
    """
    assert (num_samples_since_activated is None) == (ghost_threshold is None)
    ghost_grads = num_samples_since_activated is not None and ghost_threshold is not None
    if isinstance(activations, tuple): # for cases when the input to the autoencoder is not the same as the output
        in_acts, out_acts = activations
    else: # typically the input to the autoencoder is the same as the output
        in_acts = out_acts = activations
    f = ae.encode(in_acts)
    x_hat = ae.decode(f)
    mse_loss = t.nn.MSELoss()(
        out_acts, x_hat
    ).sqrt()

    if ghost_grads:
        # check if any neurons are dead 
        deads = (f == 0).all(dim=0).int()
        num_samples_since_activated *= deads # reset the ones that are not dead
        num_samples_since_activated += deads
        """
        From https://transformer-circuits.pub/2024/jan-update/index.html#dict-learning-resampling

        The method is to calculate an additional term that we add to the loss. This term is calculated by:

        1. Computing the reconstruction residuals and the MSE loss as normal.
        2. Computing a second forward pass of the autoencoder using just the dead neurons. 
        In this forward pass, we replace the ReLU activation function on the dead neurons
        with an exponential activation function.
            i. We determine which neurons are dead for these purposes by applying a threshold
            to the number of samples since the neuron last activated.
        3. Scaling the output of the dead neurons so that the L2 norm is 1/2 the L2 norm of the
        autoencoder residual from (1). Note that the scale factor is treated as a constant
        for gradient propagation purposes.
        4. Computing the MSE between the autoencoder residual and the output from the dead neurons.
        5. Rescaling that MSE to be equal in magnitude to the normal reconstruction loss from (1).
           The normal reconstruction loss is treated as a constant in this step for gradient propagation purposes.
        6. Adding the result to the total loss.

        This procedure is a little convoluted, but the intuition here is that we want to get a gradient signal that pushes
        the parameters of the dead neurons in the direction of explaining the autoencoder residual, as that's a promising
        place in parameter space to add more live neurons.
        """
        # 1.
        residual = in_acts - x_hat
        # 2. i
        ghosts = (num_samples_since_activated > ghost_threshold).to(f.dtype)
        if (ghosts.sum() == 0.0):
            ghost_loss = t.tensor(0.0, dtype=f.dtype)
        else:
            # 2. this is just the forward pass
            x_hat_ghost_unscaled = ae.decode(t.exp(ae.encoder(in_acts - ae.bias)) * ghosts)
            # 3.
            residual_norm = residual.norm(dim=-1, keepdim=True)
            x_hat_ghost = x_hat_ghost_unscaled * residual_norm.detach() / (x_hat_ghost_unscaled.norm(dim=-1, keepdim=True).detach() * 2)
            # 4.
            ghost_loss_unscaled = t.nn.MSELoss()(
                residual, x_hat_ghost
            ).sqrt()
            # 5.
            ghost_loss = ghost_loss_unscaled * mse_loss.detach() / ghost_loss_unscaled.detach()
    else:
        ghost_loss = t.tensor(0.0, dtype=f.dtype)

    if use_entropy:
        sparsity_loss = entropy(f)
    else:
        sparsity_loss = f.norm(p=1, dim=-1).mean()
    if separate:
        return mse_loss, sparsity_loss, ghost_loss
    else:
        return mse_loss + sparsity_penalty * sparsity_loss + ghost_loss
    
def resample_neurons(deads, activations, ae, optimizer):
    """
    resample dead neurons according to the following scheme:
    Reinitialize the decoder vector for each dead neuron to be an activation
    vector v from the dataset with probability proportional to ae's loss on v.
    Reinitialize all dead encoder vectors to be the mean alive encoder vector x 0.2.
    Reset the bias vectors for dead neurons to 0.
    Reset the Adam parameters for the dead neurons to their default values.
    """
    with t.no_grad():
        if isinstance(activations, tuple):
            in_acts, out_acts = activations
        else:
            in_acts = out_acts = activations
        in_acts = in_acts.reshape(-1, in_acts.shape[-1])
        out_acts = out_acts.reshape(-1, out_acts.shape[-1])

        # compute the loss for each activation vector
        losses = (out_acts - ae(in_acts)).norm(dim=-1)

        # resample decoder vectors for dead neurons
        indices = t.multinomial(losses, num_samples=deads.sum(), replacement=True)
        ae.decoder.weight[:,deads] = out_acts[indices].T
        ae.decoder.weight /= ae.decoder.weight.norm(dim=0, keepdim=True)

        # resample encoder vectors for dead neurons
        ae.encoder.weight[deads] = ae.encoder.weight[~deads].mean(dim=0) * 0.2

        # reset bias vectors for dead neurons
        ae.encoder.bias[deads] = 0.

        # reset Adam parameters for dead neurons
        state_dict = optimizer.state_dict()['state']
        # # encoder weight
        state_dict[1]['exp_avg'][deads] = 0.
        state_dict[1]['exp_avg_sq'][deads] = 0.
        # # encoder bias
        state_dict[2]['exp_avg'][deads] = 0.
        state_dict[2]['exp_avg_sq'][deads] = 0.


def trainSAE(
        activations, # a generator that outputs batches of activations
        activation_dim, # dimension of the activations
        dictionary_size, # size of the dictionary
        lr,
        sparsity_penalty,
        entropy=False,
        steps=None, # if None, train until activations are exhausted
        warmup_steps=1000, # linearly increase the learning rate for this many steps
        resample_steps=25000, # how often to resample dead neurons
        ghost_threshold=None, # how many steps a neuron has to be dead for it to turn into a ghost
        save_steps=None, # how often to save checkpoints
        save_dir=None, # directory for saving checkpoints
        log_steps=1000, # how often to print statistics
        device='cpu'):
    """
    Train and return a sparse autoencoder
    """
    ae = AutoEncoder(activation_dim, dictionary_size).to(device)
    alives = t.zeros(dictionary_size).bool().to(device) # which neurons are not dead?

    num_samples_since_activated = t.zeros(dictionary_size, dtype=int).to(device) if ghost_threshold is not None else None # how many samples since each neuron was last activated?

    assert not (ghost_threshold is not None and resample_steps is not None) # we can't have ghost gradients and resampling

    # set up optimizer and scheduler
    optimizer = ConstrainedAdam(ae.parameters(), ae.decoder.parameters(), lr=lr)
    def warmup_fn(step):
        if resample_steps and step % resample_steps < warmup_steps:
            return (step % resample_steps) / warmup_steps
        else:
            return 1.
    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)

    for step, acts in enumerate(tqdm(activations, total=steps)):
        if steps is not None and step >= steps:
            break

        if isinstance(acts, t.Tensor): # typical casse
            acts = acts.to(device)
        elif isinstance(acts, tuple): # for cases where the autoencoder input and output are different
            acts = tuple(a.to(device) for a in acts)

        optimizer.zero_grad()
        loss = sae_loss(acts, ae, sparsity_penalty, entropy, separate=False, num_samples_since_activated=num_samples_since_activated, ghost_threshold=ghost_threshold)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # deal with resampling neurons
        if resample_steps is not None:
            with t.no_grad():
                if isinstance(acts, tuple):
                    in_acts = acts[0]
                else:
                    in_acts = acts
                dict_acts = ae.encode(in_acts)
                alives = t.logical_or(alives, (dict_acts != 0).any(dim=0))
                if step % resample_steps == resample_steps // 2:
                    alives = t.zeros(dictionary_size).bool().to(device)
                if step % resample_steps == resample_steps - 1:
                    deads = ~alives
                    if deads.sum() > 0:
                        print(f"resampling {deads.sum().item()} dead neurons")
                        resample_neurons(deads, acts, ae, optimizer)

        # logging
        if log_steps is not None and step % log_steps == 0:
            with t.no_grad():
                mse_loss, sparsity_loss, ghost_loss = sae_loss(acts, ae, sparsity_penalty, entropy, separate=True, num_samples_since_activated=num_samples_since_activated, ghost_threshold=ghost_threshold)
                print(f"step {step} MSE loss: {mse_loss}, sparsity loss: {sparsity_loss}, ghost_loss: {ghost_loss}")
                # dict_acts = ae.encode(acts)
                # print(f"step {step} % inactive: {(dict_acts == 0).all(dim=0).sum() / dict_acts.shape[-1]}")
                # if isinstance(activations, ActivationBuffer):
                #     tokens = activations.tokenized_batch().input_ids
                #     loss_orig, loss_reconst, loss_zero = reconstruction_loss(tokens, activations.model, activations.submodule, ae)
                #     print(f"step {step} reconstruction loss: {loss_orig}, {loss_reconst}, {loss_zero}")

        # saving
        if save_steps is not None and save_dir is not None and step % save_steps == 0:
            if not os.path.exists(os.path.join(save_dir, "checkpoints")):
                os.mkdir(os.path.join(save_dir, "checkpoints"))
            t.save(
                ae.state_dict(), 
                os.path.join(save_dir, "checkpoints", f"ae_{step}.pt")
                )

    return ae