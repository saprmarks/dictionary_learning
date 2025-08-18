from datasets import load_dataset
import zstandard as zstd
import io
import json
import os
from transformers import AutoModelForCausalLM
from fractions import Fraction
import random
from transformers import AutoTokenizer
import torch as t

from .trainers.top_k import AutoEncoderTopK
from .trainers.batch_top_k import BatchTopKSAE
from .trainers.matryoshka_batch_top_k import MatryoshkaBatchTopKSAE
from .dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)


def hf_dataset_to_generator(dataset_name, split="train", streaming=True):
    dataset = load_dataset(dataset_name, split=split, streaming=streaming)

    def gen():
        for x in iter(dataset):
            yield x["text"]

    return gen()


def zst_to_generator(data_path):
    """
    Load a dataset from a .jsonl.zst file.
    The jsonl entries is assumed to have a 'text' field
    """
    compressed_file = open(data_path, "rb")
    dctx = zstd.ZstdDecompressor()
    reader = dctx.stream_reader(compressed_file)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8")

    def generator():
        for line in text_stream:
            yield json.loads(line)["text"]

    return generator()


def randomly_remove_system_prompt(
    text: str, freq: float, system_prompt: str | None = None
) -> str:
    if system_prompt and random.random() < freq:
        assert system_prompt in text
        text = text.replace(system_prompt, "")
    return text


def hf_mixed_dataset_to_generator(
    tokenizer: AutoTokenizer,
    pretrain_dataset: str = "HuggingFaceFW/fineweb",
    chat_dataset: str = "lmsys/lmsys-chat-1m",
    min_chars: int = 1,
    pretrain_frac: float = 0.9,  # 0.9 → 90 % pretrain, 10 % chat
    split: str = "train",
    streaming: bool = True,
    pretrain_key: str = "text",
    chat_key: str = "conversation",
    sequence_pack_pretrain: bool = True,
    sequence_pack_chat: bool = False,
    system_prompt_to_remove: str | None = None,
    system_prompt_removal_freq: float = 0.9,
):
    """Get a mix of pretrain and chat data at a specified ratio. By default, 90% of the data will be pretrain and 10% will be chat.

    Default datasets:
    pretrain_dataset: "HuggingFaceFW/fineweb"
    chat_dataset: "lmsys/lmsys-chat-1m"

    Note that you will have to request permission for lmsys (instant approval on HuggingFace).

    min_chars: minimum number of characters per sample. To perform sequence packing, set it to ~4x sequence length in tokens.
    Samples will be joined with the eos token.
    If it's low (like 1), each sample will just be a single row from the dataset, padded to the max length. Sometimes this will fill the context, sometimes it won't.

    Why use strings instead of tokens? Because dictionary learning expects an iterator of strings, and this is simple and good enough.

    Implicit assumption: each sample will be truncated to sequence length when tokenized.

    By default, we sequence pack the pretrain data and DO NOT sequence pack the chat data, as it would look kind of weird. The EOS token is used to separate
    user / assistant messages, not to separate conversations from different users.
    If you want to sequence pack the chat data, set sequence_pack_chat to True.

    Pretrain format will be: <bos>text<eos>text<eos>text<eos>...
    Chat format will be <formatted chat message> Optionally: <formatted chat message><formatted chat message>...

    Other parameters:
    - system_prompt_to_remove: an optional string that will be removed from the chat data with a given frequency.
        You probably want to verify that the system prompt you pass in is correct.
    - system_prompt_removal_freq: the frequency with which the system prompt will be removed

    Why? Well, we probably don't want to have 1000's of copies of the system prompt in the training dataset. But we also may not want to remove it entirely.
    And we may want to use the LLM with no system prompt when comparing between models.
    IDK, this is a complicated and annoying detail. At least this constrains the complexity to the dataset generator.
    """
    if not 0 < pretrain_frac < 1:
        raise ValueError("main_frac must be between 0 and 1 (exclusive)")

    assert min_chars > 0

    # Load both datasets as iterable streams
    pretrain_ds = iter(load_dataset(pretrain_dataset, split=split, streaming=streaming))
    chat_ds = iter(load_dataset(chat_dataset, split=split, streaming=streaming))

    # Convert the fraction to two small integers (e.g. 0.9 → 9 / 10)
    frac = Fraction(pretrain_frac).limit_denominator()
    n_pretrain = frac.numerator
    n_chat = frac.denominator - n_pretrain
    eos_token = tokenizer.eos_token

    bos_token = tokenizer.bos_token if tokenizer.bos_token else eos_token

    def gen():
        while True:
            for _ in range(n_pretrain):
                if sequence_pack_pretrain:
                    length = 0
                    samples = []
                    while length < min_chars:
                        # Add bos token to the beginning of the sample
                        sample = next(pretrain_ds)[pretrain_key]
                        samples.append(sample)
                        length += len(sample)
                    samples = bos_token + eos_token.join(samples)
                    yield samples
                else:
                    sample = bos_token + next(pretrain_ds)[pretrain_key]
                    yield sample
            for _ in range(n_chat):
                if sequence_pack_chat:
                    length = 0
                    samples = []
                    while length < min_chars:
                        sample = next(chat_ds)[chat_key]
                        # Apply chat template also includes bos token
                        sample = tokenizer.apply_chat_template(sample, tokenize=False)
                        sample = randomly_remove_system_prompt(
                            sample, system_prompt_removal_freq, system_prompt_to_remove
                        )
                        samples.append(sample)
                        length += len(sample)
                    samples = "".join(samples)
                    yield samples
                else:
                    sample = tokenizer.apply_chat_template(
                        next(chat_ds)[chat_key], tokenize=False
                    )
                    sample = randomly_remove_system_prompt(
                        sample, system_prompt_removal_freq, system_prompt_to_remove
                    )
                    yield sample

    return gen()


def hf_sequence_packing_dataset_to_generator(
    tokenizer: AutoTokenizer,
    pretrain_dataset: str = "HuggingFaceFW/fineweb",
    min_chars: int = 1,
    split: str = "train",
    streaming: bool = True,
    pretrain_key: str = "text",
    sequence_pack_pretrain: bool = True,
):
    """min_chars: minimum number of characters per sample. To perform sequence packing, set it to ~4x sequence length in tokens.
    Samples will be joined with the eos token.
    If it's low (like 1), each sample will just be a single row from the dataset, padded to the max length. Sometimes this will fill the context, sometimes it won't."""
    assert min_chars > 0

    # Load both datasets as iterable streams
    pretrain_ds = iter(load_dataset(pretrain_dataset, split=split, streaming=streaming))

    eos_token = tokenizer.eos_token

    bos_token = tokenizer.bos_token if tokenizer.bos_token else eos_token

    def gen():
        while True:
            if sequence_pack_pretrain:
                length = 0
                samples = []
                while length < min_chars:
                    # Add bos token to the beginning of the sample
                    sample = next(pretrain_ds)[pretrain_key]
                    samples.append(sample)
                    length += len(sample)
                samples = bos_token + eos_token.join(samples)
                yield samples
            else:
                sample = bos_token + next(pretrain_ds)[pretrain_key]
                yield sample

    return gen()


def simple_hf_mixed_dataset_to_generator(
    main_name: str,
    aux_name: str,
    main_frac: float = 0.9,  # 0.9 → 90 % main, 10 % aux
    split: str = "train",
    streaming: bool = True,
    main_key: str = "text",
    aux_key: str = "text",
):
    if not 0 < main_frac < 1:
        raise ValueError("main_frac must be between 0 and 1 (exclusive)")

    # Load both datasets as iterable streams
    main_ds = iter(load_dataset(main_name, split=split, streaming=streaming))
    aux_ds = iter(load_dataset(aux_name, split=split, streaming=streaming))

    # Convert the fraction to two small integers (e.g. 0.9 → 9 / 10)
    frac = Fraction(main_frac).limit_denominator()
    n_main = frac.numerator
    n_aux = frac.denominator - n_main

    def gen():
        while True:
            # Yield `n_main` items from the main dataset
            for _ in range(n_main):
                yield next(main_ds)[main_key]
            # Yield `n_aux` items from the auxiliary dataset
            for _ in range(n_aux):
                yield next(aux_ds)[aux_key]

    return gen()


def get_nested_folders(path: str) -> list[str]:
    """
    Recursively get a list of folders that contain an ae.pt file, starting the search from the given path
    """
    folder_names = []

    for root, dirs, files in os.walk(path):
        if "ae.pt" in files:
            folder_names.append(root)

    return folder_names


def load_dictionary(base_path: str, device: str) -> tuple:
    ae_path = f"{base_path}/ae.pt"
    config_path = f"{base_path}/config.json"

    with open(config_path, "r") as f:
        config = json.load(f)

    dict_class = config["trainer"]["dict_class"]

    if dict_class == "AutoEncoder":
        dictionary = AutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "GatedAutoEncoder":
        dictionary = GatedAutoEncoder.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderNew":
        dictionary = AutoEncoderNew.from_pretrained(ae_path, device=device)
    elif dict_class == "AutoEncoderTopK":
        k = config["trainer"]["k"]
        dictionary = AutoEncoderTopK.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "BatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = BatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "MatryoshkaBatchTopKSAE":
        k = config["trainer"]["k"]
        dictionary = MatryoshkaBatchTopKSAE.from_pretrained(ae_path, k=k, device=device)
    elif dict_class == "JumpReluAutoEncoder":
        dictionary = JumpReluAutoEncoder.from_pretrained(ae_path, device=device)
    else:
        raise ValueError(f"Dictionary class {dict_class} not supported")

    return dictionary, config


def get_submodule(model: AutoModelForCausalLM, layer: int):
    """Gets the residual stream submodule"""
    model_name = model.name_or_path

    if model.config.architectures[0] == "GPTNeoXForCausalLM":
        return model.gpt_neox.layers[layer]
    elif (
        model.config.architectures[0] == "Qwen2ForCausalLM"
        or model.config.architectures[0] == "Gemma2ForCausalLM"
        or model.config.architectures[0] == "Qwen3ForCausalLM"
    ):
        return model.model.layers[layer]
    else:
        raise ValueError(f"Please add submodule for model {model_name}")


def truncate_model(model: AutoModelForCausalLM, layer: int):
    """From tilde-research/activault
    https://github.com/tilde-research/activault/blob/db6d1e4e36c2d3eb4fdce79e72be94f387eccee1/pipeline/setup.py#L74
    This provides significant memory savings by deleting all layers that aren't needed for the given layer.
    You should probably test this before using it"""
    import gc

    total_params_before = sum(p.numel() for p in model.parameters())
    print(f"Model parameters before truncation: {total_params_before:,}")

    if (
        model.config.architectures[0] == "Qwen2ForCausalLM"
        or model.config.architectures[0] == "Gemma2ForCausalLM"
        or model.config.architectures[0] == "Qwen3ForCausalLM"
    ):
        removed_layers = model.model.layers[layer + 1 :]

        model.model.layers = model.model.layers[: layer + 1]

        del removed_layers
        del model.lm_head

        model.lm_head = t.nn.Identity()

    elif model.config.architectures[0] == "GPTNeoXForCausalLM":
        removed_layers = model.gpt_neox.layers[layer + 1 :]

        model.gpt_neox.layers = model.gpt_neox.layers[: layer + 1]

        del removed_layers
        del model.embed_out

        model.embed_out = t.nn.Identity()

    else:
        raise ValueError(f"Please add truncation for model {model.name_or_path}")

    total_params_after = sum(p.numel() for p in model.parameters())
    print(f"Model parameters after truncation: {total_params_after:,}")

    gc.collect()
    t.cuda.empty_cache()

    return model
