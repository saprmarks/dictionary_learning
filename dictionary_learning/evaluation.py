"""
Utilities for evaluating dictionaries on a model and dataset.
"""

import torch as t
from .buffer import ActivationBuffer, NNsightActivationBuffer
from nnsight import LanguageModel
from .config import DEBUG


def loss_recovered(
    text,  # a batch of text
    model: LanguageModel,  # an nnsight LanguageModel
    submodule,  # submodules of model
    dictionary,  # dictionaries for submodules
    max_len=None,  # max context length for loss recovered
    normalize_batch=False,  # normalize batch before passing through dictionary
    io="out",  # can be 'in', 'out', or 'in_and_out'
    tracer_args={
        "use_cache": False,
        "output_attentions": False,
    },  # minimize cache during model trace.
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
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == "in":
            x = submodule.input[0]
            if type(submodule.input.shape) == tuple:
                x = x[0]
            if normalize_batch:
                scale = (dictionary.activation_dim**0.5) / x.norm(dim=-1).mean()
                x = x * scale
        elif io == "out":
            x = submodule.output
            if type(submodule.output.shape) == tuple:
                x = x[0]
            if normalize_batch:
                scale = (dictionary.activation_dim**0.5) / x.norm(dim=-1).mean()
                x = x * scale
        elif io == "in_and_out":
            x = submodule.input[0]
            if type(submodule.input.shape) == tuple:
                x = x[0]
            print(f"x.shape: {x.shape}")
            if normalize_batch:
                scale = (dictionary.activation_dim**0.5) / x.norm(dim=-1).mean()
                x = x * scale
        else:
            raise ValueError(f"Invalid value for io: {io}")
        x = x.save()

    # pull this out so dictionary can be written without FakeTensor (top_k needs this)
    x_hat = dictionary(x.view(-1, x.shape[-1])).view(x.shape).to(model.dtype)

    # intervene with `x_hat`
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == "in":
            x = submodule.input[0]
            if normalize_batch:
                scale = (dictionary.activation_dim**0.5) / x.norm(dim=-1).mean()
                x_hat = x_hat / scale
            if type(submodule.input.shape) == tuple:
                submodule.input[0][:] = x_hat
            else:
                submodule.input = x_hat
        elif io == "out":
            x = submodule.output
            if normalize_batch:
                scale = (dictionary.activation_dim**0.5) / x.norm(dim=-1).mean()
                x_hat = x_hat / scale
            if type(submodule.output.shape) == tuple:
                submodule.output = (x_hat,)
            else:
                submodule.output = x_hat
        elif io == "in_and_out":
            x = submodule.input[0]
            if normalize_batch:
                scale = (dictionary.activation_dim**0.5) / x.norm(dim=-1).mean()
                x_hat = x_hat / scale
            submodule.output = x_hat
        else:
            raise ValueError(f"Invalid value for io: {io}")

        logits_reconstructed = model.output.save()
    logits_reconstructed = logits_reconstructed.value

    # logits when replacing component activations with zeros
    with model.trace(text, **tracer_args, invoker_args=invoker_args):
        if io == "in":
            x = submodule.input[0]
            if type(submodule.input.shape) == tuple:
                submodule.input[0][:] = t.zeros_like(x[0])
            else:
                submodule.input = t.zeros_like(x)
        elif io in ["out", "in_and_out"]:
            x = submodule.output
            if type(submodule.output.shape) == tuple:
                submodule.output[0][:] = t.zeros_like(x[0])
            else:
                submodule.output = t.zeros_like(x)
        else:
            raise ValueError(f"Invalid value for io: {io}")

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

    if isinstance(text, t.Tensor):
        tokens = text
    else:
        try:
            tokens = input[1]["input_ids"]
        except:
            tokens = input[1]["input"]

    # compute losses
    losses = []
    if hasattr(model, "tokenizer") and model.tokenizer is not None:
        loss_kwargs = {"ignore_index": model.tokenizer.pad_token_id}
    else:
        loss_kwargs = {}
    for logits in [logits_original, logits_reconstructed, logits_zero]:
        loss = t.nn.CrossEntropyLoss(**loss_kwargs)(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1)
        )
        losses.append(loss)

    return tuple(losses)


def evaluate(
    dictionary,  # a dictionary
    activations,  # a generator of activations; if an ActivationBuffer, also compute loss recovered
    max_len=128,  # max context length for loss recovered
    batch_size=128,  # batch size for loss recovered
    io="out",  # can be 'in', 'out', or 'in_and_out'
    normalize_batch=False,  # normalize batch before passing through dictionary
    tracer_args={
        "use_cache": False,
        "output_attentions": False,
    },  # minimize cache during model trace.
    device="cpu",
):
    with t.no_grad():

        out = {}  # dict of results

        try:
            x = next(activations).to(device)
            if normalize_batch:
                x = x / x.norm(dim=-1).mean() * (dictionary.activation_dim**0.5)

        except StopIteration:
            raise StopIteration(
                "Not enough activations in buffer. Pass a buffer with a smaller batch size or more data."
            )

        x_hat, f = dictionary(x, output_features=True)
        l2_loss = t.linalg.norm(x - x_hat, dim=-1).mean()
        l1_loss = f.norm(p=1, dim=-1).mean()
        l0 = (f != 0).float().sum(dim=-1).mean()
        frac_alive = (
            t.flatten(f, start_dim=0, end_dim=1).any(dim=0).sum() / dictionary.dict_size
        )

        # cosine similarity between x and x_hat
        x_normed = x / t.linalg.norm(x, dim=-1, keepdim=True)
        x_hat_normed = x_hat / t.linalg.norm(x_hat, dim=-1, keepdim=True)
        cossim = (x_normed * x_hat_normed).sum(dim=-1).mean()

        # l2 ratio
        l2_ratio = (t.linalg.norm(x_hat, dim=-1) / t.linalg.norm(x, dim=-1)).mean()

        # compute variance explained
        total_variance = t.var(x, dim=0).sum()
        residual_variance = t.var(x - x_hat, dim=0).sum()
        frac_variance_explained = 1 - residual_variance / total_variance

        # Equation 10 from https://arxiv.org/abs/2404.16014
        x_hat_norm_squared = t.linalg.norm(x_hat, dim=-1, ord=2) ** 2
        x_dot_x_hat = (x * x_hat).sum(dim=-1)
        relative_reconstruction_bias = x_hat_norm_squared.mean() / x_dot_x_hat.mean()

        out["l2_loss"] = l2_loss.item()
        out["l1_loss"] = l1_loss.item()
        out["l0"] = l0.item()
        out["frac_alive"] = frac_alive.item()
        out["frac_variance_explained"] = frac_variance_explained.item()
        out["cossim"] = cossim.item()
        out["l2_ratio"] = l2_ratio.item()
        out["relative_reconstruction_bias"] = relative_reconstruction_bias.item()

        if not isinstance(activations, (ActivationBuffer, NNsightActivationBuffer)):
            return out

        # compute loss recovered
        loss_original, loss_reconstructed, loss_zero = loss_recovered(
            activations.text_batch(batch_size=batch_size),
            activations.model,
            activations.submodule,
            dictionary,
            max_len=max_len,
            normalize_batch=normalize_batch,
            io=io,
            tracer_args=tracer_args,
        )
        frac_recovered = (loss_reconstructed - loss_zero) / (loss_original - loss_zero)

        out["loss_original"] = loss_original.item()
        out["loss_reconstructed"] = loss_reconstructed.item()
        out["loss_zero"] = loss_zero.item()
        out["frac_recovered"] = frac_recovered.item()

        return out
