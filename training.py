"""
Training dictionaries
"""

import os
from typing import Optional, Union

import torch as t
from tqdm import tqdm

from .dictionary import AbstractAutoEncoder, AutoEncoder, GatedAutoEncoder

EPS = 1e-8


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


def entropy(p: t.Tensor, eps=EPS):
    p_sum = p.sum(dim=-1, keepdim=True)
    # epsilons for numerical stability
    p_normed = p / (p_sum + eps)
    p_log = t.log(p_normed + eps)
    ent = -(p_normed * p_log)

    # Zero out the entropy where p_sum is zero
    ent = t.where(p_sum > 0, ent, t.zeros_like(ent))

    return ent.sum(dim=-1).mean()


def sae_loss(
    activations: Union[t.Tensor, tuple[t.Tensor, t.Tensor]],
    autoencoder: AutoEncoder,
    sparsity_penalty: float = 0,
    use_entropy: bool = False,
    output_all_losses: bool = False,
    num_samples_since_activated: Optional[t.Tensor] = None,
    ghost_threshold: Optional[float] = None,
):
    """
    Compute the loss of an autoencoder on some activations
    If separate is True, return the MSE loss, the sparsity loss, and the ghost loss separately
    If num_samples_since_activated is not None, update it in place
    If ghost_threshold is not None, use it to do ghost grads
    """
    if isinstance(
        activations, tuple
    ):  # for cases when the input to the autoencoder is not the same as the output
        in_acts, out_acts = activations
    else:  # typically the input to the autoencoder is the same as the output
        in_acts = out_acts = activations

    ghost_grads = False
    if ghost_threshold is not None:
        if num_samples_since_activated is None:
            raise ValueError("num_samples_since_activated must be provided for ghost grads")
        ghost_mask = num_samples_since_activated > ghost_threshold
        if ghost_mask.sum() > 0:  # if there are dead neurons
            ghost_grads = True
        else:
            ghost_loss = None

    if not ghost_grads:  # if we're not doing ghost grads
        x_hat, f = autoencoder(in_acts, output_features=True)
        mse_loss = t.linalg.norm(out_acts - x_hat, dim=-1).mean()

    else:  # if we're doing ghost grads
        x_hat, x_ghost, f = autoencoder(in_acts, output_features=True, ghost_mask=ghost_mask)
        residual = out_acts - x_hat
        mse_loss = t.linalg.norm(residual, dim=-1).mean()
        x_ghost = (
            x_ghost
            * residual.norm(dim=-1, keepdim=True).detach()
            / (2 * x_ghost.norm(dim=-1, keepdim=True).detach() + EPS)
        )
        ghost_loss = t.linalg.norm(residual.detach() - x_ghost, dim=-1).mean()

    if (
        num_samples_since_activated is not None
    ):  # update the number of samples since each neuron was last activated
        deads = (f == 0).all(dim=0)
        num_samples_since_activated.copy_(t.where(deads, num_samples_since_activated + 1, 0))

    if use_entropy:
        sparsity_loss = entropy(f)
    else:
        sparsity_loss = f.norm(p=1, dim=-1).mean()

    if output_all_losses:
        return mse_loss, sparsity_loss, ghost_loss
    elif ghost_grads and ghost_loss is not None:
        return (
            mse_loss
            + sparsity_penalty * sparsity_loss
            + ghost_loss * (mse_loss.detach() / (ghost_loss.detach() + EPS))
        )
    else:
        return mse_loss + sparsity_penalty * sparsity_loss


def gated_sae_loss(
    activations: t.Tensor,
    gated_autoencoder: GatedAutoEncoder,
    output_all_losses: bool = False,
    l1_coef: float = 1.0,
) -> Union[t.Tensor, tuple[t.Tensor, t.Tensor, t.Tensor]]:
    """
    Compute the loss of a gated autoencoder on some activations
    """

    (
        reconstructed_activations,
        _features,
        active_features_pre_binarisation,
        _active_features,
        _feature_magnitudes,
    ) = gated_autoencoder(activations, output_features=True)

    # We’ll use the reconstruction from the baseline forward pass to train
    # the magnitudes encoder and decoder.
    mse_reconstruction_loss = t.linalg.norm(
        activations - reconstructed_activations, dim=-1
    ).mean()

    # We apply a L1 penalty on the gated encoder activations (pre-binarising,
    # post-ReLU) to incentivise them to be sparse
    relued_gates = t.relu(active_features_pre_binarisation)  # [batch_size, num_features]
    gating_sparsity_loss = l1_coef * relued_gates.norm(p=1, dim=-1).mean()

    # Apply a reconstruction loss on the gating encoder to encourage it to
    # reconstruct well. We apply a stop gradient for the decoding.
    with t.no_grad():
        gating_only_reconstruction = gated_autoencoder.decode(relued_gates)
    gating_reconstruction_loss = t.linalg.norm(
        activations - gating_only_reconstruction, dim=-1
    ).mean()

    if output_all_losses:
        return mse_reconstruction_loss, gating_sparsity_loss, gating_reconstruction_loss
    else:
        return mse_reconstruction_loss + gating_sparsity_loss + gating_reconstruction_loss


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
        if deads.sum() == 0:
            return
        if isinstance(activations, tuple):
            in_acts, out_acts = activations
        else:
            in_acts = out_acts = activations
        in_acts = in_acts.reshape(-1, in_acts.shape[-1])
        out_acts = out_acts.reshape(-1, out_acts.shape[-1])

        # compute the loss for each activation vector
        losses = (out_acts - ae(in_acts)).norm(dim=-1)

        # sample input to create encoder/decoder weights from
        n_resample = min([deads.sum(), losses.shape[0]])
        indices = t.multinomial(losses, num_samples=n_resample, replacement=False)
        sampled_vecs = activations[indices]

        # get norm of the living neurons
        alive_norm = ae.encoder.weight[~deads].norm(dim=-1).mean()

        # resample first n_resample dead neurons
        deads[deads.nonzero()[n_resample:]] = False
        ae.encoder.weight[deads] = sampled_vecs * alive_norm * 0.2
        ae.decoder.weight[:, deads] = (
            sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True)
        ).T
        ae.encoder.bias[deads] = 0.0

        # reset Adam parameters for dead neurons
        state_dict = optimizer.state_dict()["state"]
        ## encoder weight
        state_dict[1]["exp_avg"][deads] = 0.0
        state_dict[1]["exp_avg_sq"][deads] = 0.0
        ## encoder bias
        state_dict[2]["exp_avg"][deads] = 0.0
        state_dict[2]["exp_avg_sq"][deads] = 0.0
        ## decoder weight
        state_dict[3]["exp_avg"][:, deads] = 0.0
        state_dict[3]["exp_avg_sq"][:, deads] = 0.0


def trainSAE(
    activations,  # a generator that outputs batches of activations
    activation_dim,  # dimension of the activations
    dictionary_size: int,  # size of the dictionary
    lr,
    sparsity_penalty,
    entropy=False,
    steps=None,  # if None, train until activations are exhausted
    warmup_steps=1000,  # linearly increase the learning rate for this many steps
    resample_steps=None,  # how often to resample dead neurons
    ghost_threshold=None,  # how many steps a neuron has to be dead for it to turn into a ghost
    save_steps=None,  # how often to save checkpoints
    save_dir=None,  # directory for saving checkpoints
    log_steps=1000,  # how often to print statistics
    device="cpu",
) -> AbstractAutoEncoder:
    """
    Train and return a sparse autoencoder
    """
    ae = AutoEncoder(activation_dim, dictionary_size).to(device)
    num_samples_since_activated = t.zeros(
        dictionary_size, dtype=t.int32, device=device
    )  # how many samples since each neuron was last activated?

    # set up optimizer and scheduler
    optimizer = ConstrainedAdam(ae.parameters(), ae.decoder.parameters(), lr=lr)
    if resample_steps is None:

        def warmup_fn(step):
            return min(step / warmup_steps, 1.0)

    else:

        def warmup_fn(step):
            return min((step % resample_steps) / warmup_steps, 1.0)

    scheduler = t.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)

    for step, acts in enumerate(tqdm(activations, total=steps)):
        if steps is not None and step >= steps:
            break

        if isinstance(acts, t.Tensor):  # typical case
            acts = acts.to(device)
        elif isinstance(
            acts, tuple
        ):  # for cases where the autoencoder input and output are different
            acts = tuple(a.to(device) for a in acts)

        optimizer.zero_grad()
        # computing the sae_loss also updates num_samples_since_activated in place
        loss = sae_loss(
            acts,
            ae,
            sparsity_penalty,
            use_entropy=entropy,
            num_samples_since_activated=num_samples_since_activated,
            ghost_threshold=ghost_threshold,
        )
        assert isinstance(loss, t.Tensor)

        loss.backward()
        optimizer.step()
        scheduler.step()

        # deal with resampling neurons
        if resample_steps is not None and step % resample_steps == 0:
            # resample neurons who've been dead for the last resample_steps / 2 steps
            resample_neurons(
                num_samples_since_activated > resample_steps / 2, acts, ae, optimizer
            )

        # logging
        if log_steps is not None and step % log_steps == 0:
            with t.no_grad():
                losses = sae_loss(
                    acts,
                    ae,
                    sparsity_penalty,
                    entropy,
                    output_all_losses=True,
                    num_samples_since_activated=num_samples_since_activated,
                    ghost_threshold=ghost_threshold,
                )
                if ghost_threshold is None:
                    mse_loss, sparsity_loss, _ = losses
                    print(f"step {step} MSE loss: {mse_loss}, sparsity loss: {sparsity_loss}")
                else:
                    mse_loss, sparsity_loss, ghost_loss = losses
                    print(
                        f"step {step} MSE loss: {mse_loss}, sparsity loss: {sparsity_loss}, ghost_loss: {ghost_loss}"
                    )
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
            t.save(ae.state_dict(), os.path.join(save_dir, "checkpoints", f"ae_{step}.pt"))

    return ae
