import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import Dataset
from nnsight import LanguageModel
from typing import Tuple, List
import numpy as np
import os
from tqdm.auto import tqdm
import json

from .config import DEBUG

if DEBUG:
    tracer_kwargs = {"scan": True, "validate": True}
else:
    tracer_kwargs = {"scan": False, "validate": False}


class ActivationShard:
    def __init__(self, store_dir: str, shard_idx: int):
        self.shard_file = os.path.join(store_dir, f"shard_{shard_idx}.memmap")
        with open(self.shard_file.replace(".memmap", ".meta"), "r") as f:
            self.shape = tuple(json.load(f)["shape"])
        self.activations = np.memmap(
            self.shard_file, dtype=np.float32, mode="r", shape=self.shape
        )

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, *indices):
        return th.tensor(self.activations[*indices], dtype=th.float32)


class ActivationCache:
    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        self.config = json.load(open(os.path.join(store_dir, "config.json"), "r"))
        self.shards = [
            ActivationShard(store_dir, i) for i in range(self.config["shard_count"])
        ]
        self._range_to_shard_idx = np.cumsum([0] + [s.shape[0] for s in self.shards])

    def __len__(self):
        return self.config["total_size"]

    def __getitem__(self, index: int):
        shard_idx = np.searchsorted(self._range_to_shard_idx, index, side="right") - 1
        offset = index - self._range_to_shard_idx[shard_idx]
        shard = self.shards[shard_idx]
        return shard[offset]

    @staticmethod
    def get_activations(submodule: nn.Module, io: str):
        if io == "in":
            return submodule.input[0]
        else:
            return submodule.output[0]

    @staticmethod
    def collate_store_shards(
        store_dirs: Tuple[str],
        shard_count: int,
        activation_cache: List[th.Tensor],
        submodule_names: Tuple[str],
        shuffle_shards: bool = True,
        io: str = "out",
    ):
        for i, name in enumerate(submodule_names):
            activations = th.cat(
                activation_cache[i], dim=0
            )  # (N x B x T) x D (N = number of batches per shard)
            print(f"Storing activation shard ({activations.shape}) for {name} {io}")
            if shuffle_shards:
                idx = np.random.permutation(activations.shape[0])
                activations = activations[idx]
            # use memmap to store activations
            memmap_file = os.path.join(store_dirs[i], f"shard_{shard_count}.memmap")
            memmap_file_meta = memmap_file.replace(".memmap", ".meta")
            memmap = np.memmap(
                memmap_file,
                dtype=np.float32,
                mode="w+",
                shape=(activations.shape[0], activations.shape[1]),
            )
            memmap[:] = activations.numpy()
            memmap.flush()
            with open(memmap_file_meta, "w") as f:
                json.dump({"shape": list(activations.shape)}, f)
            del memmap

    @th.no_grad()
    @staticmethod
    def collect(
        data: Dataset,
        submodules: Tuple[nn.Module],
        submodule_names: Tuple[str],
        model: LanguageModel,
        store_dir: str,
        batch_size: int = 64,
        context_len: int = 128,
        shard_size: int = 10**6,
        d_model: int = 1024,
        shuffle_shards: bool = False,
        io: str = "out",
        num_workers: int = 8,
        max_total_tokens: int = 10**8,
        last_submodule: nn.Module = None,
    ):

        dataloader = DataLoader(data, batch_size=batch_size, num_workers=num_workers)

        activation_cache = [[] for _ in submodules]
        store_dirs = [
            os.path.join(store_dir, f"{submodule_names[i]}_{io}")
            for i in range(len(submodules))
        ]
        for store_dir in store_dirs:
            os.makedirs(store_dir, exist_ok=True)
        total_size = 0
        current_size = 0
        shard_count = 0
        for batch in tqdm(dataloader, desc="Collecting activations"):
            tokens = model.tokenizer(
                batch,
                max_length=context_len,
                truncation=True,
                return_tensors="pt",
                padding=True,
            ).to(model.device)
            attention_mask = tokens["attention_mask"]
            with model.trace(
                tokens,
                **tracer_kwargs,
            ):
                for i, submodule in enumerate(submodules):
                    local_activations = (
                        ActivationCache.get_activations(submodule, io)
                        .reshape(-1, d_model)
                        .save()
                    )  # (B x T) x D
                    activation_cache[i].append(local_activations)

                if last_submodule is not None:
                    last_submodule.output.stop()

            for i in range(len(submodules)):
                activation_cache[i][-1] = (
                    activation_cache[i][-1]
                    .value[attention_mask.reshape(-1).bool()]
                    .cpu()
                    .to(th.float32)
                )  # remove padding tokens

            current_size += activation_cache[0][-1].shape[0]

            if current_size > shard_size:
                ActivationCache.collate_store_shards(
                    store_dirs,
                    shard_count,
                    activation_cache,
                    submodule_names,
                    shuffle_shards,
                    io,
                )
                shard_count += 1

                total_size += current_size
                current_size = 0
                activation_cache = [[] for _ in submodules]

            if total_size > max_total_tokens:
                print(f"Max total tokens reached. Stopping collection.")
                break

        if current_size > 0:
            ActivationCache.collate_store_shards(
                store_dirs,
                shard_count,
                activation_cache,
                submodule_names,
                shuffle_shards,
                io,
            )

        # store configs
        for i, store_dir in enumerate(store_dirs):
            with open(os.path.join(store_dir, "config.json"), "w") as f:
                json.dump(
                    {
                        "batch_size": batch_size,
                        "context_len": context_len,
                        "shard_size": shard_size,
                        "d_model": d_model,
                        "shuffle_shards": shuffle_shards,
                        "io": io,
                        "total_size": total_size,
                        "shard_count": shard_count,
                    },
                    f,
                )
        print(f"Finished collecting activations. Total size: {total_size}")


class PairedActivationCache:
    def __init__(self, store_dir_1: str, store_dir_2: str):
        self.activation_cache_1 = ActivationCache(store_dir_1)
        self.activation_cache_2 = ActivationCache(store_dir_2)
        assert len(self.activation_cache_1) == len(self.activation_cache_2)

    def __len__(self):
        return len(self.activation_cache_1)

    def __getitem__(self, index: int):
        return th.stack(
            (self.activation_cache_1[index], self.activation_cache_2[index]), dim=0
        )


class ActivationCacheTuple:
    def __init__(self, *store_dirs: str):
        self.activation_caches = [
            ActivationCache(store_dir) for store_dir in store_dirs
        ]
        assert len(self.activation_caches) > 0
        for i in range(1, len(self.activation_caches)):
            assert len(self.activation_caches[i]) == len(self.activation_caches[0])

    def __len__(self):
        return len(self.activation_caches[0])

    def __getitem__(self, index: int):
        return th.stack([cache[index] for cache in self.activation_caches], dim=0)
