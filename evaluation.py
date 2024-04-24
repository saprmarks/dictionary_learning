"""
Utilities for evaluating dictionaries on a model and dataset.
"""

import torch as t
from .buffer import ActivationBuffer
from nnsight import LanguageModel
from .config import DEBUG


def loss_recovered(
    text,  # a batch of text
    model: LanguageModel,  # an nnsight LanguageModel
    submodule,  # submodules of model
    dictionary,  # dictionaries for submodules
    max_len=None,  # max context length for loss recovered
    io="out",  # can be 'in', 'out', or 'in_to_out'
):
    """
    How much of the model's loss is recovered by replacing the component output
    with the reconstruction by the autoencoder?
    """

    if max_len is None:
        invoker_args = {}
    else:
        invoker_args = {"truncation": True, "max_length": max_len}

    # unmodified logits
    with model.trace(text, invoker_args=invoker_args):
        logits_original = model.output.save()
    logits_original = logits_original.value

    # logits when replacing component activations with reconstruction by autoencoder
    with model.trace(text, invoker_args=invoker_args):
        if io == 'in':
            x = submodule.input[0]
            if isinstance(x, tuple):
                submodule.input[0][:] = dictionary(x[0])
            else:
                submodule.input = dictionary(x)
        elif io == 'out':
            x = submodule.output
            if isinstance(x, tuple):
                submodule.output[0][:] = dictionary(x[0])
            else:
                submodule.output = dictionary(x)
        
        logits_reconstructed = model.output.save()
    logits_reconstructed = logits_reconstructed.value

    # logits when replacing component activations with zeros
    with model.trace(text, invoker_args=invoker_args):
        if io == 'in':
            x = submodule.input[0]
            if isinstance(x, tuple):
                submodule.input[0][:] = t.zeros_like(x[0])
            else:
                submodule.input = t.zeros_like(x)
        elif io == 'out':
            x = submodule.output
            if isinstance(x, tuple):
                submodule.output[0][:] = t.zeros_like(x[0])
            else:
                submodule.output = t.zeros_like(x)
        
        input = model.input.save()
        logits_zero = model.output.save()
    logits_zero = logits_zero.value

    # get everything into the right format
    try:
        logits_original = logits_original.logits
        logits_reconstructed = logits_reconstructed.logits
        logits_zero = logits_zero.logits
    except:
        pass

    try:
        tokens = input[1]['input_ids']
    except:
        tokens = input[1]['input']

    # compute losses
    losses = []
    for logits in [logits_original, logits_reconstructed, logits_zero]:
        loss = t.nn.CrossEntropyLoss(ignore_index=model.tokenizer.pad_token_id)(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
        )
        losses.append(loss)

    return tuple(losses)


def evaluate(
    dictionary,  # a dictionary
    activations, # a generator of activations; if an ActivationBuffer, also compute loss recovered
    max_len=128,  # max context length for loss recovered
    batch_size=128,  # batch size for loss recovered
    io="out",  # can be 'in', 'out', or 'in_to_out'
    device="cpu",
):
    with t.no_grad():

        out = {}  # dict of results

        try:
            x = next(activations).to(device)
        except StopIteration:
            raise StopIteration(
                "Not enough activations in buffer. Pass a buffer with a smaller batch size or more data."
            )
        
        x_hat, f = dictionary(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        l0 = (f != 0).float().sum(dim=-1).mean()
        frac_alive = (f != 0).float().mean(dim=-1).mean()

        #compute variance explained
        total_variance = t.var(x, dim=0).sum()
        residual_variance = t.var(x - x_hat, dim=0).sum()
        frac_variance_explained = (1 - residual_variance / total_variance)

        out["l2_loss"] = l2_loss.item()
        out["l1_loss"] = l1_loss.item()
        out["l0"] = l0.item()
        out["frac_alive"] = frac_alive.item()
        out["frac_variance_explained"] = frac_variance_explained.item()

        if not isinstance(activations, ActivationBuffer):
            return out

        # compute loss recovered
        loss_original, loss_reconstructed, loss_zero = loss_recovered(
            activations.text_batch(batch_size=batch_size),
            activations.model,
            activations.submodule,
            dictionary,
            max_len=max_len,
            io=io,
        )
        frac_recovered = (loss_reconstructed - loss_zero) / (loss_original - loss_zero)
        
        out["loss_original"] = loss_original.item()
        out["loss_reconstructed"] = loss_reconstructed.item()
        out["loss_zero"] = loss_zero.item()
        out["frac_recovered"] = frac_recovered.item()

        return out
