import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc
from tqdm import tqdm
import contextlib


class EarlyStopException(Exception):
    """Custom exception for stopping model forward pass early."""

    pass


def collect_activations(
    model: AutoModelForCausalLM,
    submodule: t.nn.Module,
    inputs_BL: dict[str, t.Tensor],
    use_no_grad: bool = True,
) -> t.Tensor:
    """
    Registers a forward hook on the submodule to capture the residual (or hidden)
    activations. We then raise an EarlyStopException to skip unneeded computations.

    Args:
        model: The model to run.
        submodule: The submodule to hook into.
        inputs_BL: The inputs to the model.
        use_no_grad: Whether to run the forward pass within a `t.no_grad()` context. Defaults to True.
    """
    activations_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal activations_BLD
        # For many models, the submodule outputs are a tuple or a single tensor:
        # If "outputs" is a tuple, pick the relevant item:
        #   e.g. if your layer returns (hidden, something_else), you'd do outputs[0]
        # Otherwise just do outputs
        if isinstance(outputs, tuple):
            activations_BLD = outputs[0]
        else:
            activations_BLD = outputs

        raise EarlyStopException("Early stopping after capturing activations")

    handle = submodule.register_forward_hook(gather_target_act_hook)

    # Determine the context manager based on the flag
    context_manager = t.no_grad() if use_no_grad else contextlib.nullcontext()

    try:
        # Use the selected context manager
        with context_manager:
            _ = model(**inputs_BL)
    except EarlyStopException:
        pass
    except Exception as e:
        print(f"Unexpected error during forward pass: {str(e)}")
        raise
    finally:
        handle.remove()

    if activations_BLD is None:
        # This should ideally not happen if the hook worked and EarlyStopException was raised,
        # but handle it just in case.
        raise RuntimeError(
            "Failed to collect activations. The hook might not have run correctly."
        )

    return activations_BLD


class ActivationBuffer:
    """
    Implements a buffer of activations. The buffer stores activations from a model,
    yields them in batches, and refreshes them when the buffer is less than half full.

    max_activation_norm_multiple: remove all activations with norm greater than median norm * max_activation_norm_multiple. 10 is a good default.
    This is useful for models like Qwen which have random, unpredictable high norm activation sinks which reduce training effectiveness.
    """

    def __init__(
        self,
        data,  # generator which yields text data
        model: AutoModelForCausalLM,  # Language Model from which to extract activations
        submodule,  # submodule of the model from which to extract activations
        d_submodule=None,  # submodule dimension; if None, try to detect automatically
        io="out",  # can be 'in' or 'out'; whether to extract input or output activations
        n_ctxs=3e4,  # approximate number of contexts to store in the buffer
        ctx_len=128,  # length of each context
        refresh_batch_size=512,  # size of batches in which to process the data when adding to buffer
        out_batch_size=8192,  # size of batches in which to yield activations
        device="cpu",  # device on which to store the activations
        remove_bos: bool = False,
        add_special_tokens: bool = True,
        max_activation_norm_multiple: int | None = None,
    ):
        if io not in ["in", "out"]:
            raise ValueError("io must be either 'in' or 'out'")

        if d_submodule is None:
            try:
                if io == "in":
                    d_submodule = submodule.in_features
                else:
                    d_submodule = submodule.out_features
            except:
                raise ValueError(
                    "d_submodule cannot be inferred and must be specified directly"
                )
        self.activations = t.empty(0, d_submodule, device=device, dtype=model.dtype)
        self.read = t.zeros(0).bool()

        self.data = data
        self.model = model
        self.submodule = submodule
        self.d_submodule = d_submodule
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.activation_buffer_size = n_ctxs * ctx_len
        self.refresh_batch_size = refresh_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
        self.add_special_tokens = add_special_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model.name_or_path)
        self.remove_bos = remove_bos
        self.remove_high_norm = max_activation_norm_multiple

        if remove_bos and self.tokenizer.bos_token_id is None:
            print(
                "\n\n\nWARNING: remove_bos is True but tokenizer does not have a bos token. We are removing the first non-pad token instead. Don't use sequence packing.\n\n\n"
            )

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.activation_buffer_size // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[
                t.randperm(len(unreads), device=unreads.device)[: self.out_batch_size]
            ]
            self.read[idxs] = True
            return self.activations[idxs]

    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.refresh_batch_size
        try:
            return [next(self.data) for _ in range(batch_size)]
        except StopIteration:
            raise StopIteration("End of data stream reached")

    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.tokenizer(
            texts,
            return_tensors="pt",
            max_length=self.ctx_len,
            padding=True,
            truncation=True,
            add_special_tokens=self.add_special_tokens,
        ).to(self.device)

    def refresh(self):
        gc.collect()
        t.cuda.empty_cache()
        self.activations = self.activations[~self.read]

        current_idx = len(self.activations)
        new_activations = t.empty(
            self.activation_buffer_size,
            self.d_submodule,
            device=self.device,
            dtype=self.model.dtype,
        )

        new_activations[: len(self.activations)] = self.activations
        self.activations = new_activations

        # Optional progress bar when filling buffer. At larger models / buffer sizes (e.g. gemma-2-2b, 1M tokens on a 4090) this can take a couple minutes.
        # pbar = tqdm(total=self.activation_buffer_size, initial=current_idx, desc="Refreshing activations")

        while current_idx < self.activation_buffer_size:
            with t.no_grad():
                input = self.tokenized_batch()
                hidden_states = collect_activations(self.model, self.submodule, input)
            mask = input["attention_mask"] != 0
            if self.remove_bos:
                if self.tokenizer.bos_token_id is not None:
                    bos_mask = input["input_ids"] == self.tokenizer.bos_token_id
                    mask = mask & ~bos_mask
                else:
                    # some models (like Qwen) don't have a bos token, so we need to remove the first non-pad token
                    assert mask.dim() == 2, "expected shape (batch_size, seq_len)"
                    first_one = (mask.to(t.int64).cumsum(dim=1) == 1) & mask
                    mask = mask & ~first_one

            if self.remove_high_norm is not None:
                # some models (like Qwen) have random high norm activation sinks which reduce training effectiveness
                norms_BL = hidden_states.norm(dim=-1)
                median_norm = norms_BL.median()
                norm_mask = norms_BL > median_norm * self.remove_high_norm
                mask = mask & ~norm_mask

            hidden_states = hidden_states[mask]

            remaining_space = self.activation_buffer_size - current_idx
            assert remaining_space > 0
            hidden_states = hidden_states[:remaining_space]

            self.activations[current_idx : current_idx + len(hidden_states)] = (
                hidden_states.to(self.device)
            )
            current_idx += len(hidden_states)

            # pbar.update(len(hidden_states))

        # pbar.close()
        self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    @property
    def config(self):
        return {
            "d_submodule": self.d_submodule,
            "io": self.io,
            "n_ctxs": self.n_ctxs,
            "ctx_len": self.ctx_len,
            "refresh_batch_size": self.refresh_batch_size,
            "out_batch_size": self.out_batch_size,
            "device": self.device,
        }
