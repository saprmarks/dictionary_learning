"""Copyright (2025) Tilde Research Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import io
import json
import os
import random
import signal
import sys
import time
import warnings
from multiprocessing import Process, Queue, Value
from typing import Optional

import einops
import aiohttp
import boto3
import torch
import torch.nn as nn
import multiprocessing as mp
import warnings
import logging

logger = logging.getLogger(__name__)

# Constants for file sizes
KB = 1024
MB = KB * KB

# Cache directory constants
OUTER_CACHE_DIR = "cache"
INNER_CACHE_DIR = "cache"
BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "main")


def _metadata_path(run_name):
    """Generate the metadata file path for a given run name."""
    return f"{run_name}/metadata.json"


def _statistics_path(run_name):
    """Generate the statistics file path for a given run name."""
    return f"{run_name}/statistics.json"


async def download_chunks(session, url, total_size, chunk_size):
    """Download file chunks asynchronously with retries."""
    tries_left = 5
    while tries_left > 0:
        chunks = [
            (i, min(i + chunk_size - 1, total_size - 1))
            for i in range(0, total_size, chunk_size)
        ]
        tasks = [
            asyncio.create_task(request_chunk(session, url, start, end))
            for start, end in chunks
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        results = []
        retry = False
        for response in responses:
            if isinstance(response, Exception):
                logger.error(f"Error occurred: {response}")
                logger.error(
                    f"Session: {session}, URL: {url}, Tries left: {tries_left}"
                )
                tries_left -= 1
                retry = True
                break
            else:
                results.append(response)

        if not retry:
            return results

    return None


async def request_chunk(session, url, start, end):
    """Request a specific chunk of a file."""
    headers = {"Range": f"bytes={start}-{end}"}
    try:
        async with session.get(url, headers=headers) as response:
            response.raise_for_status()
            return start, await response.read()
    except Exception as e:
        return e


def download_loop(*args):
    """Run the asynchronous download loop."""
    asyncio.run(_async_download(*args))


def compile(byte_buffers, shuffle=True, seed=None, return_ids=False):
    """Compile downloaded chunks into a tensor."""
    combined_bytes = b"".join(
        chunk for _, chunk in sorted(byte_buffers, key=lambda x: x[0])
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # n = np.frombuffer(combined_bytes, dtype=np.float16)
        # t = torch.from_numpy(n)
        # t = torch.frombuffer(combined_bytes, dtype=dtype) # torch.float32
        buffer = io.BytesIO(combined_bytes)
        t = torch.load(buffer)
        if (
            isinstance(t, dict) and "states" in t and not return_ids
        ):  # backward compatibility
            t = t["states"]  # ignore input_ids
        buffer.close()

    if shuffle and not return_ids:
        t = shuffle_megabatch_tokens(t, seed)

    return t


def shuffle_megabatch_tokens(t, seed=None):
    """
    Shuffle within a megabatch (across batches and sequences), using each token as the unit of shuffling.

    Args:
    t (torch.Tensor): Input tensor of shape (batch_size * batches_per_file, sequence_length, d_in + 1)
    seed (int): Seed for the random number generator

    Returns:
    torch.Tensor: Shuffled tensor of the same shape as input
    """
    original_shape = (
        t.shape
    )  # (batch_size * batches_per_file, sequence_length, d_in + 1)

    total_tokens = (
        original_shape[0] * original_shape[1]
    )  # reshape to (total_tokens, d_in + 1)
    t_reshaped = t.reshape(total_tokens, -1)

    rng = torch.Generator()
    if seed is not None:
        rng.manual_seed(seed)

    shuffled_indices = torch.randperm(total_tokens, generator=rng)
    t_shuffled = t_reshaped[shuffled_indices]

    t = t_shuffled.reshape(original_shape)  # revert

    return t


def write_tensor(t, buffer, writeable_tensors, readable_tensors, ongoing_downloads):
    """Write a tensor to the shared buffer."""
    idx = writeable_tensors.get(block=True)
    if isinstance(buffer[0], SharedBuffer):
        buffer[idx].states.copy_(t["states"])
        buffer[idx].input_ids.copy_(t["input_ids"])
    else:
        buffer[idx] = t

    readable_tensors.put(idx, block=True)
    with ongoing_downloads.get_lock():
        ongoing_downloads.value -= 1


async def _async_download(
    buffer,
    file_index,
    s3_paths,
    stop,
    readable_tensors,
    writeable_tensors,
    ongoing_downloads,
    concurrency,
    bytes_per_file,
    chunk_size,
    shuffle,
    seed,
    return_ids,
):
    """Asynchronously download and process files from S3."""
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        while file_index.value < len(s3_paths) and not stop.value:
            with ongoing_downloads.get_lock():
                ongoing_downloads.value += 1
            with file_index.get_lock():
                url = s3_paths[file_index.value]
                file_index.value += 1
            bytes_results = await download_chunks(
                session, url, bytes_per_file, chunk_size
            )
            if bytes_results is not None:
                try:
                    t = compile(bytes_results, shuffle, seed, return_ids)
                    write_tensor(
                        t,
                        buffer,
                        writeable_tensors,
                        readable_tensors,
                        ongoing_downloads,
                    )
                except Exception as e:
                    logger.error(f"Exception while downloading: {e}")
                    logger.error(f"Failed URL: {url}")
                    stop.value = True  # Set stop flag
                    break  # Exit the loop
            else:
                logger.error(f"Failed to download URL: {url}")
                with ongoing_downloads.get_lock():
                    ongoing_downloads.value -= 1


class S3RCache:
    """A cache that reads data from Amazon S3."""

    @classmethod
    def from_credentials(
        self, aws_access_key_id, aws_secret_access_key, *args, **kwargs
    ):
        s3_client = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            endpoint_url=os.environ.get("S3_ENDPOINT_URL"),
        )
        return S3RCache(s3_client, *args, **kwargs)

    def __init__(
        self,
        s3_client,
        s3_prefix,
        bucket_name=BUCKET_NAME,
        device="cpu",
        concurrency=100,
        chunk_size=MB * 16,
        buffer_size=2,
        shuffle=True,
        preserve_file_order=False,
        seed=42,
        paths=None,
        n_workers=1,
        return_ids=False,
    ) -> None:
        """Initialize S3 cache."""
        ensure_spawn_context()

        # Configure S3 client with correct signature version
        self.s3_client = (
            boto3.client(
                "s3",
                region_name="eu-north1",  # Make sure this matches your bucket region
                config=boto3.session.Config(signature_version="s3v4"),
            )
            if s3_client is None
            else s3_client
        )

        self.s3_prefix = s3_prefix
        self.bucket_name = bucket_name
        self.device = device
        self.concurrency = concurrency
        self.chunk_size = chunk_size
        self.buffer_size = buffer_size
        self.shuffle = shuffle
        self.preserve_file_order = preserve_file_order
        self.seed = seed
        self.return_ids = return_ids

        random.seed(self.seed)
        torch.manual_seed(self.seed)  # unclear if this has effect
        # but we drill down the seed to download loop anyway

        self.paths = paths
        self._s3_paths = self._list_s3_files()
        if isinstance(self.s3_prefix, list):
            target_prefix = self.s3_prefix[0]
        else:
            target_prefix = self.s3_prefix
        response = self.s3_client.get_object(
            Bucket=bucket_name, Key=_metadata_path(target_prefix)
        )
        content = response["Body"].read()
        self.metadata = json.loads(content)
        # self.metadata["bytes_per_file"] = 1612711320
        self._activation_dtype = eval(self.metadata["dtype"])

        self._running_processes = []
        self.n_workers = n_workers

        self.readable_tensors = Queue(maxsize=self.buffer_size)
        self.writeable_tensors = Queue(maxsize=self.buffer_size)

        for i in range(self.buffer_size):
            self.writeable_tensors.put(i)

        if self.return_ids:
            self.buffer = [
                SharedBuffer(
                    self.metadata["shape"],
                    self.metadata["input_ids_shape"],
                    self._activation_dtype,
                )
                for _ in range(self.buffer_size)
            ]
            for shared_buffer in self.buffer:
                shared_buffer.share_memory()
        else:
            self.buffer = torch.empty(
                (self.buffer_size, *self.metadata["shape"]),
                dtype=self._activation_dtype,
            ).share_memory_()

        self._stop = Value("b", False)
        self._file_index = Value("i", 0)
        self._ongoing_downloads = Value("i", 0)

        signal.signal(signal.SIGTERM, self._catch_stop)
        signal.signal(signal.SIGINT, self._catch_stop)

        self._initial_file_index = 0

    @property
    def current_file_index(self):
        return self._file_index.value

    def set_file_index(self, index):
        self._initial_file_index = index

    def _catch_stop(self, *args, **kwargs):
        logger.info("cleaning up before process is killed")
        self._stop_downloading()
        sys.exit(0)

    def sync(self):
        self._s3_paths = self._list_s3_files()

    def _reset(self):
        self._file_index.value = self._initial_file_index
        self._ongoing_downloads.value = 0
        self._stop.value = False

        while not self.readable_tensors.empty():
            self.readable_tensors.get()

        while not self.writeable_tensors.empty():
            self.writeable_tensors.get()
        for i in range(self.buffer_size):
            self.writeable_tensors.put(i)

    def _list_s3_files(self):
        """List and prepare all data files from one or more S3 prefixes."""
        paths = []
        combined_metadata = None
        combined_config = None

        # Handle single prefix case for backward compatibility
        prefixes = (
            [self.s3_prefix] if isinstance(self.s3_prefix, str) else self.s3_prefix
        )

        # Process each prefix
        for prefix in prefixes:
            # Get metadata for this prefix
            response = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=_metadata_path(prefix)
            )
            metadata = json.loads(response["Body"].read())

            # Get config for this prefix
            try:
                config_response = self.s3_client.get_object(
                    Bucket=self.bucket_name,
                    Key=f"{'/'.join(prefix.split('/')[:-1])}/cfg.json",
                )
                config = json.loads(config_response["Body"].read())
            except Exception as e:
                logger.warning(
                    f"Warning: Could not load config for prefix {prefix}: {e}"
                )
                config = {}

            # Initialize combined metadata and config from first prefix
            if combined_metadata is None:
                combined_metadata = metadata.copy()
                combined_config = config.copy()
                # Initialize accumulation fields
                combined_config["total_tokens"] = 0
                combined_config["n_total_files"] = 0
                combined_config["batches_processed"] = 0
            else:
                # Verify metadata compatibility
                if metadata["shape"][1:] != combined_metadata["shape"][1:]:
                    raise ValueError(
                        f"Incompatible shapes between datasets: {metadata['shape']} vs {combined_metadata['shape']}"
                    )
                if metadata["dtype"] != combined_metadata["dtype"]:
                    raise ValueError(f"Incompatible dtypes between datasets")

            # Accumulate config fields
            combined_config["total_tokens"] += config.get("total_tokens", 0)
            combined_config["n_total_files"] += config.get("n_total_files", 0)
            combined_config["batches_processed"] += config.get("batches_processed", 0)

            # List files for this prefix
            paginator = self.s3_client.get_paginator("list_objects_v2")
            page_iterator = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            prefix_paths = []
            for page in page_iterator:
                if "Contents" not in page:
                    continue

                for obj in page["Contents"]:
                    if (
                        obj["Key"] != _metadata_path(prefix)
                        and obj["Key"] != _statistics_path(prefix)
                        and not obj["Key"].endswith("cfg.json")
                    ):
                        url = self.s3_client.generate_presigned_url(
                            "get_object",
                            Params={"Bucket": self.bucket_name, "Key": obj["Key"]},
                            ExpiresIn=604700,
                        )
                        prefix_paths.append(url)

            paths.extend(prefix_paths)

        # Store the combined metadata and config
        self.metadata = combined_metadata
        self.config = combined_config  # Store combined config for potential later use

        if self.preserve_file_order:
            # chronological upload order
            return sorted(paths)
        else:
            # shuffle the file order
            random.shuffle(paths)
            return paths

    def __iter__(self):
        self._reset()

        if self._running_processes:
            raise ValueError(
                "Cannot iterate over cache a second time while it is downloading"
            )

        if len(self._s3_paths) > self._initial_file_index:
            while len(self._running_processes) < self.n_workers:
                p = Process(
                    target=download_loop,
                    args=(
                        self.buffer,
                        self._file_index,
                        self._s3_paths[
                            self._initial_file_index :
                        ],  # Start from the initial index
                        self._stop,
                        self.readable_tensors,
                        self.writeable_tensors,
                        self._ongoing_downloads,
                        self.concurrency,
                        self.metadata["bytes_per_file"],
                        self.chunk_size,
                        self.shuffle,
                        self.seed,
                        self.return_ids,
                    ),
                )
                p.start()
                self._running_processes.append(p)
                time.sleep(0.75)

        return self

    def _next_tensor(self):
        try:
            idx = self.readable_tensors.get(block=True)
            if self.return_ids:
                t = {
                    "states": self.buffer[idx].states.clone().detach(),
                    "input_ids": self.buffer[idx].input_ids.clone().detach(),
                }
            else:
                t = self.buffer[idx].clone().detach()

            self.writeable_tensors.put(idx, block=True)
            return t
        except Exception as e:
            logger.error(f"exception while iterating: {e}")
            self._stop_downloading()
            raise StopIteration

    def __next__(self):
        while (
            self._file_index.value < len(self._s3_paths)
            or not self.readable_tensors.empty()
            or self._ongoing_downloads.value > 0
        ):
            return self._next_tensor()

        if self._running_processes:
            self._stop_downloading()
        raise StopIteration

    def finalize(self):
        self._stop_downloading()

    def _stop_downloading(self):
        logger.info("stopping workers...")
        self._file_index.value = len(self._s3_paths)
        self._stop.value = True

        while not all([not p.is_alive() for p in self._running_processes]):
            if not self.readable_tensors.empty():
                self.readable_tensors.get()

            if not self.writeable_tensors.full():
                self.writeable_tensors.put(0)

            time.sleep(0.25)

        for p in self._running_processes:
            p.join()  # still join to make sure all resources are cleaned up

        self._ongoing_downloads.value = 0
        self._running_processes = []


"""
tl;dr of why we need this:
shared memory is handled differently for nested structures -- see buffer intiialization
we can initialize a dict with two tensors with shared memory, and these tensors themselves are shared but NOT the dict
hence writing to buffer[idx] in write_tensor will not actually write to self.buffer[idx], which _next_tensor uses
(possibly a better fix, but for now this works)
"""


class SharedBuffer(nn.Module):
    def __init__(self, shape, input_ids_shape, dtype):
        super().__init__()
        self.states = nn.Parameter(torch.ones(shape, dtype=dtype), requires_grad=False)
        self.input_ids = nn.Parameter(
            torch.ones(input_ids_shape, dtype=torch.int64), requires_grad=False
        )

    def forward(self):
        return {"states": self.states, "input_ids": self.input_ids}


### mini-helper for multiprocessing
def ensure_spawn_context():
    """
    Ensures multiprocessing uses 'spawn' context if not already set.
    Returns silently if already set to 'spawn'.
    Issues warning if unable to set to 'spawn'.
    """
    if mp.get_start_method(allow_none=True) != "spawn":
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            warnings.warn(
                "Multiprocessing start method is not 'spawn'. This may cause issues."
            )


def create_s3_client(
    access_key_id: Optional[str] = None,
    secret_access_key: Optional[str] = None,
    endpoint_url: Optional[str] = None,
) -> boto3.client:
    """Create an S3 client configured for S3-compatible storage services.

    This function creates a boto3 S3 client with optimized settings for reliable
    data transfer. It supports both direct credential passing and environment
    variable configuration.

    Args:
        access_key_id: S3 access key ID. If None, reads from AWS_ACCESS_KEY_ID env var
        secret_access_key: S3 secret key. If None, reads from AWS_SECRET_ACCESS_KEY env var
        endpoint_url: S3-compatible storage service endpoint URL

    Returns:
        boto3.client: Configured S3 client with optimized settings

    Environment Variables:
        - AWS_ACCESS_KEY_ID: S3 access key ID (if not provided as argument)
        - AWS_SECRET_ACCESS_KEY: S3 secret key (if not provided as argument)

    Example:
        ```python
        # Using environment variables
        s3_client = create_s3_client()

        # Using explicit credentials
        s3_client = create_s3_client(
            access_key_id="your_key",
            secret_access_key="your_secret",
            endpoint_url="your_endpoint_url"
        )
        ```

    Note:
        The client is configured with path-style addressing and S3v4 signatures
        for maximum compatibility with S3-compatible storage services.
    """
    access_key_id = access_key_id or os.environ.get("AWS_ACCESS_KEY_ID")
    secret_access_key = secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY")
    endpoint_url = endpoint_url or os.environ.get("S3_ENDPOINT_URL")

    if not access_key_id or not secret_access_key:
        raise ValueError(
            "S3 credentials must be provided either through arguments or "
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables"
        )

    if not endpoint_url:
        raise ValueError(
            "S3 endpoint URL must be provided either through arguments or "
            "S3_ENDPOINT_URL environment variable"
        )

    session = boto3.session.Session()
    return session.client(
        service_name="s3",
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        endpoint_url=endpoint_url,
        use_ssl=True,
        verify=True,
        config=boto3.session.Config(
            s3={"addressing_style": "path"},
            signature_version="s3v4",
            # Advanced configuration options (currently commented out):
            # retries=dict(
            #     max_attempts=3,  # Number of retry attempts
            #     mode='adaptive'  # Adds exponential backoff
            # ),
            # max_pool_connections=20,  # Limits concurrent connections
            # connect_timeout=60,  # Connection timeout in seconds
            # read_timeout=300,    # Read timeout in seconds
            # tcp_keepalive=True,  # Enable TCP keepalive
        ),
    )


class ActivaultS3ActivationBuffer:
    def __init__(
        self,
        cache: S3RCache,
        batch_size: int = 8192,
        device: str = "cpu",
        io: str = "out",
    ):
        self.cache = iter(cache)  # Make sure it's an iterator
        self.batch_size = batch_size
        self.device = device
        self.io = io

        self.states = None  # Shape: [N, D]
        self.read_mask = None  # Shape: [N]
        self.refresh()  # Load the first batch

    def __iter__(self):
        return self

    def __next__(self):
        with torch.no_grad():
            if (~self.read_mask).sum() < self.batch_size:
                self.refresh()

            if self.states is None or self.states.shape[0] == 0:
                raise StopIteration

            unreads = (~self.read_mask).nonzero().squeeze()
            if unreads.ndim == 0:
                unreads = unreads.unsqueeze(0)
            selected = unreads[
                torch.randperm(len(unreads), device=self.device)[: self.batch_size]
            ]
            self.read_mask[selected] = True
            return self.states[selected]

    def refresh(self):
        try:
            next_batch = next(self.cache)  # dict with "states" key
        except StopIteration:
            self.states = None
            self.read_mask = None
            return

        states = next_batch["states"].to(self.device)  # [B, L, D]
        flat_states = einops.rearrange(states, "b l d -> (b l) d").contiguous()
        self.states = flat_states
        self.read_mask = torch.zeros(
            flat_states.shape[0], dtype=torch.bool, device=self.device
        )

    def close(self):
        if hasattr(self.cache, "finalize"):
            self.cache.finalize()
        elif hasattr(self.cache, "close"):
            self.cache.close()


if __name__ == "__main__":
    device = "cuda"
    sae_batch_size = 2048
    io = "out"

    # example activault usage

    BUCKET_NAME = os.environ.get("S3_BUCKET_NAME", "main")
    s3_prefix = ["mistral.8b.fineweb/blocks.9.hook_resid_post"]
    cache = S3RCache.from_credentials(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        s3_prefix=s3_prefix,
        bucket_name=BUCKET_NAME,
        device=device,
        buffer_size=2,
        return_ids=True,
        shuffle=True,
        n_workers=2,
    )

    s3_buffer = ActivaultS3ActivationBuffer(
        cache, batch_size=sae_batch_size, device=device, io=io
    )
