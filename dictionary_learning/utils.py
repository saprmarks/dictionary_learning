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
