"""
Utilities for evaluating dictionaries on a model and dataset.
"""

import matplotlib.pyplot as plt
import torch as t

from nnsight import LanguageModel

from .training import sae_loss
from .config import DEBUG

if DEBUG:
    tracer_kwargs = {'scan' : True, 'validate' : True}
else:
    tracer_kwargs = {'scan' : False, 'validate' : False}


def loss_recovered(
    text,  # a batch of text
    model: LanguageModel,  # an nnsight LanguageModel
    submodules,  # submodules of model
    dictionaries,  # dictionaries for submodules
    max_len=None,  # max context length for loss recovered
    io="out",  # can be 'in', 'out', or 'in_to_out'
    pct=False,  # return pct recovered; if False, return losses
):
    """
    How much of the model's loss is recovered by replacing the component output
    with the reconstruction by the autoencoder?
    """
    # figure out which submodules output tuples
    is_tuple = {}
    with model.trace('_'):
        if io == 'out':
            for submodule in submodules:
                is_tuple[submodule] = type(submodule.output.shape) == tuple
        else:
            for submodule in submodules:
                is_tuple[submodule] = type(submodule.input.shape) == tuple

    if max_len is None:
        invoker_args = {}
    else:
        invoker_args = {"truncation": True, "max_length": max_len}

    # unmodified logits
    with t.no_grad(), model.trace(text, **tracer_kwargs, invoker_args=invoker_args):
        output = model.output.save()
    try:
        logits_original = output.value.logits
    except:
        logits_original = output.value
    
    # logits when replacing component output with reconstruction by autoencoder
    with t.no_grad(), model.trace(text, **tracer_kwargs, invoker_args=invoker_args):
        for submodule, dictionary in zip(submodules, dictionaries):
            if io == "in":
                if is_tuple[submodule]:
                    submodule.input[0][:] = dictionary(submodule.input[0]) # TODO: Fix
                else:
                    submodule.input = dictionary(submodule.input)
            elif io == "out":
                if is_tuple[submodule]:
                    submodule.output[0][:] = dictionary(submodule.output[0]) # TODO: Fix
                else:
                    submodule.output = dictionary(submodule.output)
            elif io == "in_to_out":
                if is_tuple[submodule]:
                    submodule.output[0][:] = dictionary(submodule.input[0]) # TODO: Fix
                else:
                    submodule.output = dictionary(submodule.input)
            else:
                raise ValueError(f"invalid io: {io}")
            
        output = model.output.save()

    try:
        logits_reconstructed = output.value.logits
    except:
        logits_reconstructed = output.value

    # logits when zero ablating components
    with t.no_grad(), model.trace(text, **tracer_kwargs, invoker_args=invoker_args):
        for submodule in submodules:
            if io == "in":
                if is_tuple[submodule]:
                    submodule.input[0][:] = t.zeros_like(submodule.input[0]) # TODO: Fix
                else:
                    submodule.input = t.zeros_like(submodule.input)
            elif io == "out":
                if is_tuple[submodule]:
                    submodule.output[0][:] = t.zeros_like(submodule.output[0]) # TODO: Fix
                else:
                    submodule.output = t.zeros_like(submodule.output)
            elif io == "in_to_out":
                if is_tuple[submodule]:
                    submodule.output[0][:] = t.zeros_like(submodule.input[0]) # TODO: Fix
                else:
                    submodule.output = t.zeros_like(submodule.input)
            else:
                raise ValueError(f"invalid io: {io}")
        input = model.input.save()
        output = model.output.save()

    try:
        logits_zero = output.value.logits
    except:
        logits_zero = output.value

    try:
        tokens = input[1]['input_ids']
    except:
        tokens = input[1].value['input']
    
    losses = []
    for logits in [logits_original, logits_reconstructed, logits_zero]:
        loss = t.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
        ).item()
        losses.append(loss)

    if pct:
        return (losses[1] - losses[2]) / (losses[0] - losses[2])
    else:
        return tuple(losses)


def evaluate(
    model: LanguageModel,  # a nnsight LanguageModel
    submodule,  # a submodule of model
    dictionary,  # a dictionary
    activations,  # an ActivationBuffer
    max_len=None,  # max context length for loss recovered
    batch_size=None,  # batch size for loss recovered
    entropy=False,  # whether to use entropy regularization
    hist_save_path=None,  # path for saving histograms
    hist_title=None,  # title for histograms
    io="out",  # can be 'in', 'out', or 'in_to_out'
    device="cpu",
):
    with t.no_grad():

        out = {}  # dict of results

        acts = next(activations).to(device)

        # compute reconstruction (L2) loss and sparsity loss
        mse_loss, sparsity_loss = sae_loss(
            acts, dictionary, sparsity_penalty=None, use_entropy=entropy, separate=True
        )
        out["mse_loss"] = mse_loss.item() ** 2  # / acts.norm(dim=-1).mean().item() ** 2
        out["sparsity_loss"] = sparsity_loss.item()

        # compute variance explained
        total_variance = t.var(acts, dim=0).sum()
        residual_variance = t.var(acts - dictionary(acts), dim=0).sum()
        out["variance_explained"] = (1 - residual_variance / total_variance).item()

        # compute mean L0 norm and percentage of neurons alive
        features = dictionary.encode(acts)
        actives = features != 0
        out["l0"] = actives.float().sum(dim=-1).mean().item()
        alives = actives.any(dim=0)
        out["percent_alive"] = alives.float().mean().item()

        # compute histogram if needed
        if hist_save_path is not None:
            freqs = actives.float().mean(dim=0)
            plt.figure()
            plt.hist(freqs.cpu(), bins=t.logspace(-5, 0, 100))
            plt.xscale("log")
            plt.title(hist_title)
            plt.savefig(hist_save_path)
            plt.close()

        # compute loss recovered
        loss_original, loss_reconstructed, loss_zero = loss_recovered(
            activations.text_batch(batch_size=batch_size),
            model,
            [submodule],
            [dictionary],
            max_len=max_len,
            io=io,
            pct=False,
        )
        out["loss_original"] = loss_original
        out["loss_reconstructed"] = loss_reconstructed
        out["loss_zero"] = loss_zero
        out["percent_recovered"] = (loss_reconstructed - loss_zero) / (
            loss_original - loss_zero
        )

        return out
