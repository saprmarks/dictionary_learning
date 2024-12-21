import torch as t
from nnsight import LanguageModel
import argparse
import itertools
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Optional, Type, Any
from enum import Enum

from training import trainSAE
from trainers.standard import StandardTrainer
from trainers.top_k import TopKTrainer, AutoEncoderTopK
from trainers.gdm import GatedSAETrainer
from trainers.p_anneal import PAnnealTrainer
from trainers.jumprelu import JumpReluTrainer
from utils import hf_dataset_to_generator
from buffer import ActivationBuffer
from dictionary import AutoEncoder, GatedAutoEncoder, AutoEncoderNew, JumpReluAutoEncoder
from evaluation import evaluate
import utils as utils


class TrainerType(Enum):
    STANDARD = "standard"
    STANDARD_NEW = "standard_new"
    TOP_K = "top_k"
    BATCH_TOP_K = "batch_top_k"
    GATED = "gated"
    P_ANNEAL = "p_anneal"
    JUMP_RELU = "jump_relu"


@dataclass
class LLMConfig:
    llm_batch_size: int
    context_length: int
    sae_batch_size: int
    dtype: t.dtype


@dataclass
class SparsityPenalties:
    standard: list[float]
    p_anneal: list[float]
    gated: list[float]


# TODO: Move all of these to a config file
num_tokens = 50_000_000
eval_num_inputs = 1_000
random_seeds = [0]
expansion_factors = [8]

# note: learning rate is not used for topk
learning_rates = [3e-4]

LLM_CONFIG = {
    "EleutherAI/pythia-70m-deduped": LLMConfig(
        llm_batch_size=512, context_length=128, sae_batch_size=4096, dtype=t.float32
    ),
    "google/gemma-2-2b": LLMConfig(
        llm_batch_size=32, context_length=128, sae_batch_size=2048, dtype=t.bfloat16
    ),
}


# NOTE: In the current setup, the length of each sparsity penalty and target_l0 should be the same
SPARSITY_PENALTIES = {
    "EleutherAI/pythia-70m-deduped": SparsityPenalties(
        standard=[0.01, 0.05, 0.075, 0.1, 0.125, 0.15],
        p_anneal=[0.02, 0.03, 0.035, 0.04, 0.05, 0.075],
        gated=[0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
    ),
    "google/gemma-2-2b": SparsityPenalties(
        standard=[0.025, 0.035, 0.04, 0.05, 0.06, 0.07],
        p_anneal=[-1] * 6,
        gated=[-1] * 6,
    ),
}


TARGET_L0s = [20, 40, 80, 160, 320, 640]


@dataclass
class BaseTrainerConfig:
    activation_dim: int
    dict_size: int
    seed: int
    device: str
    layer: str
    lm_name: str
    submodule_name: str
    trainer: Type[Any]
    dict_class: Type[Any]
    wandb_name: str
    steps: Optional[int] = None


@dataclass
class WarmupConfig:
    warmup_steps: int = 1000
    resample_steps: Optional[int] = None


@dataclass
class StandardTrainerConfig(BaseTrainerConfig, WarmupConfig):
    lr: float
    l1_penalty: float


@dataclass
class StandardNewTrainerConfig(BaseTrainerConfig, WarmupConfig):
    lr: float
    l1_penalty: float


@dataclass
class PAnnealTrainerConfig(BaseTrainerConfig, WarmupConfig):
    lr: float
    initial_sparsity_penalty: float
    sparsity_function: str = "Lp^p"
    p_start: float = 1.0
    p_end: float = 0.2
    anneal_start: int = 10000
    anneal_end: Optional[int] = None
    sparsity_queue_length: int = 10
    n_sparsity_updates: int = 10


@dataclass
class TopKTrainerConfig(BaseTrainerConfig):
    k: int
    auxk_alpha: float = 1 / 32
    decay_start: int = 24000
    threshold_beta: float = 0.999


@dataclass
class GatedTrainerConfig(BaseTrainerConfig, WarmupConfig):
    lr: float
    l1_penalty: float


@dataclass
class JumpReluTrainerConfig(BaseTrainerConfig):
    lr: float
    target_l0: int
    sparsity_penalty: float = 1.0
    bandwidth: float = 0.001


def get_trainer_configs(
    architectures: list[str],
    learning_rate: float,
    sparsity_index: int,
    seed: int,
    activation_dim: int,
    dict_size: int,
    model_name: str,
    device: str,
    layer: str,
    submodule_name: str,
    steps: int,
) -> list[dict]:
    trainer_configs = []

    base_config = {
        "activation_dim": activation_dim,
        "dict_size": dict_size,
        "seed": seed,
        "device": device,
        "layer": layer,
        "lm_name": model_name,
        "submodule_name": submodule_name,
    }

    if TrainerType.P_ANNEAL.value in architectures:
        config = PAnnealTrainerConfig(
            **base_config,
            trainer=PAnnealTrainer,
            dict_class=AutoEncoder,
            lr=learning_rate,
            initial_sparsity_penalty=SPARSITY_PENALTIES[model_name].p_anneal[sparsity_index],
            steps=steps,
            wandb_name=f"PAnnealTrainer-{model_name}-{submodule_name}",
        )
        trainer_configs.append(asdict(config))

    if TrainerType.STANDARD.value in architectures:
        config = StandardTrainerConfig(
            **base_config,
            trainer=StandardTrainer,
            dict_class=AutoEncoder,
            lr=learning_rate,
            l1_penalty=SPARSITY_PENALTIES[model_name].standard[sparsity_index],
            wandb_name=f"StandardTrainer-{model_name}-{submodule_name}",
        )
        trainer_configs.append(asdict(config))

    if TrainerType.STANDARD_NEW.value in architectures:
        config = StandardNewTrainerConfig(
            **base_config,
            trainer=StandardTrainer,
            dict_class=AutoEncoderNew,
            lr=learning_rate,
            l1_penalty=SPARSITY_PENALTIES[model_name].standard[sparsity_index],
            wandb_name=f"StandardTrainerNew-{model_name}-{submodule_name}",
        )
        trainer_configs.append(asdict(config))

    if TrainerType.TOP_K.value in architectures:
        config = TopKTrainerConfig(
            **base_config,
            trainer=TopKTrainer,
            dict_class=AutoEncoderTopK,
            k=TARGET_L0s[sparsity_index],
            steps=steps,
            wandb_name=f"TopKTrainer-{model_name}-{submodule_name}",
        )
        trainer_configs.append(asdict(config))

    if TrainerType.GATED.value in architectures:
        config = GatedTrainerConfig(
            **base_config,
            trainer=GatedSAETrainer,
            dict_class=GatedAutoEncoder,
            lr=learning_rate,
            l1_penalty=SPARSITY_PENALTIES[model_name].gated[sparsity_index],
            wandb_name=f"GatedTrainer-{model_name}-{submodule_name}",
        )
        trainer_configs.append(asdict(config))

    if TrainerType.JUMP_RELU.value in architectures:
        config = JumpReluTrainerConfig(
            **base_config,
            trainer=JumpReluTrainer,
            dict_class=JumpReluAutoEncoder,
            lr=learning_rate,
            target_l0=TARGET_L0s[sparsity_index],
            wandb_name=f"JumpReluTrainer-{model_name}-{submodule_name}",
        )
        trainer_configs.append(asdict(config))

    return trainer_configs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="where to store sweep")
    parser.add_argument("--use_wandb", action="store_true", help="use wandb logging")
    parser.add_argument("--dry_run", action="store_true", help="dry run sweep")
    parser.add_argument(
        "--layers", type=int, nargs="+", required=True, help="layers to train SAE on"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="which language model to use",
    )
    parser.add_argument(
        "--architectures",
        type=str,
        nargs="+",
        choices=[e.value for e in TrainerType],
        required=True,
        help="which SAE architectures to train",
    )
    args = parser.parse_args()
    return args


def run_sae_training(
    model_name: str,
    layer: int,
    save_dir: str,
    device: str,
    architectures: list,
    num_tokens: int,
    random_seeds: list[int],
    expansion_factors: list[float],
    learning_rates: list[float],
    dry_run: bool = False,
    use_wandb: bool = False,
    save_checkpoints: bool = False,
    buffer_scaling_factor: int = 20,
):
    # model and data parameters
    context_length = LLM_CONFIG[model_name]["context_length"]

    llm_batch_size = LLM_CONFIG[model_name]["llm_batch_size"]
    sae_batch_size = LLM_CONFIG[model_name]["sae_batch_size"]
    dtype = LLM_CONFIG[model_name]["dtype"]

    num_contexts_per_sae_batch = sae_batch_size // context_length
    buffer_size = num_contexts_per_sae_batch * buffer_scaling_factor

    # sae training parameters
    # random_seeds = t.arange(10).tolist()

    num_sparsities = len(TARGET_L0s)
    sparsity_indices = t.arange(num_sparsities).tolist()

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train

    if save_checkpoints:
        # Creates checkpoints at 0.1%, 0.316%, 1%, 3.16%, 10%, 31.6%, 100% of training
        desired_checkpoints = t.logspace(-3, 0, 7).tolist()
        desired_checkpoints = [0.0] + desired_checkpoints[:-1]
        desired_checkpoints.sort()
        print(f"desired_checkpoints: {desired_checkpoints}")

        save_steps = [int(steps * step) for step in desired_checkpoints]
        save_steps.sort()
        print(f"save_steps: {save_steps}")
    else:
        save_steps = None

    log_steps = 100  # Log the training on wandb
    if not use_wandb:
        log_steps = None

    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)
    submodule = utils.get_submodule(model, layer)
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    activation_dim = model.config.hidden_size

    generator = hf_dataset_to_generator("monology/pile-uncopyrighted")

    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=buffer_size,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io=io,
        d_submodule=activation_dim,
        device=device,
    )

    # create the list of configs
    trainer_configs = []

    for seed, sparsity_index, expansion_factor, learning_rate in itertools.product(
        random_seeds, sparsity_indices, expansion_factors, learning_rates
    ):
        dict_size = int(expansion_factor * activation_dim)
        trainer_configs.extend(
            get_trainer_configs(
                architectures,
                learning_rate,
                sparsity_index,
                seed,
                activation_dim,
                dict_size,
                model_name,
                device,
                submodule_name,
                steps,
            )
        )

    print(f"len trainer configs: {len(trainer_configs)}")
    save_dir = f"{save_dir}/{submodule_name}"

    if not dry_run:
        # actually run the sweep
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            use_wandb=use_wandb,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
        )


@t.no_grad()
def eval_saes(
    model_name: str,
    ae_paths: list[str],
    n_inputs: int,
    device: str,
    overwrite_prev_results: bool = False,
    transcoder: bool = False,
) -> dict:
    if transcoder:
        io = "in_and_out"
    else:
        io = "out"

    context_length = LLM_CONFIG[model_name]["context_length"]
    llm_batch_size = LLM_CONFIG[model_name]["llm_batch_size"]
    loss_recovered_batch_size = llm_batch_size // 5
    sae_batch_size = loss_recovered_batch_size * context_length
    dtype = LLM_CONFIG[model_name]["dtype"]

    model = LanguageModel(model_name, dispatch=True, device_map=device)
    model = model.to(dtype=dtype)

    buffer_size = n_inputs
    io = "out"
    n_batches = n_inputs // loss_recovered_batch_size

    generator = hf_dataset_to_generator("monology/pile-uncopyrighted")

    input_strings = []
    for i, example in enumerate(generator):
        input_strings.append(example)
        if i > n_inputs * 5:
            break

    eval_results = {}

    for ae_path in ae_paths:
        output_filename = f"{ae_path}/eval_results.json"
        if not overwrite_prev_results:
            if os.path.exists(output_filename):
                print(f"Skipping {ae_path} as eval results already exist")
                continue

        dictionary, config = utils.load_dictionary(ae_path, device)
        dictionary = dictionary.to(dtype=model.dtype)

        layer = config["trainer"]["layer"]
        submodule = utils.get_submodule(model, layer)

        activation_dim = config["trainer"]["activation_dim"]

        activation_buffer = ActivationBuffer(
            iter(input_strings),
            model,
            submodule,
            n_ctxs=buffer_size,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            out_batch_size=sae_batch_size,
            io=io,
            d_submodule=activation_dim,
            device=device,
        )

        eval_results = evaluate(
            dictionary,
            activation_buffer,
            context_length,
            loss_recovered_batch_size,
            io=io,
            device=device,
            n_batches=n_batches,
        )

        hyperparameters = {
            "n_inputs": n_inputs,
            "context_length": context_length,
        }
        eval_results["hyperparameters"] = hyperparameters

        print(eval_results)

        with open(output_filename, "w") as f:
            json.dump(eval_results, f)

    # return the final eval_results for testing purposes
    return eval_results


if __name__ == "__main__":
    """python pythia.py --save_dir ./run2 --model_name EleutherAI/pythia-70m-deduped --layers 3 --architectures standard standard_new top_k gated --use_wandb
    python pythia.py --save_dir ./run3 --model_name google/gemma-2-2b --layers 12 --architectures standard top_k --use_wandb
    python pythia.py --save_dir ./jumprelu --model_name EleutherAI/pythia-70m-deduped --layers 3 --architectures jump_relu --use_wandb"""
    args = get_args()

    device = "cuda:0"

    for layer in args.layers:
        run_sae_training(
            model_name=args.model_name,
            layer=layer,
            save_dir=args.save_dir,
            device=device,
            architectures=args.architectures,
            num_tokens=num_tokens,
            random_seeds=random_seeds,
            expansion_factors=expansion_factors,
            learning_rates=learning_rates,
            dry_run=args.dry_run,
            use_wandb=args.use_wandb,
        )

    ae_paths = utils.get_nested_folders(args.save_dir)

    eval_saes(args.model_name, ae_paths, eval_num_inputs, device)
