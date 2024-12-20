import torch as t
from nnsight import LanguageModel
import argparse
import itertools
import os
import json

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainers.top_k import TrainerTopK, AutoEncoderTopK
from dictionary_learning.trainers.gdm import GatedSAETrainer
from dictionary_learning.trainers.p_anneal import PAnnealTrainer
from dictionary_learning.trainers.jumprelu import JumpReluTrainer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder, GatedAutoEncoder, AutoEncoderNew, JumpReluAutoEncoder
from dictionary_learning.evaluation import evaluate
import dictionary_learning.utils as utils


DEVICE = "cuda:0"

LLM_CONFIG = {
    "EleutherAI/pythia-70m-deduped": {
        "llm_batch_size": 512,
        "context_length": 128,
        "sae_batch_size": 4096,
        "dtype": t.float32,
    },
    "google/gemma-2-2b": {
        "llm_batch_size": 32,
        "context_length": 128,
        "sae_batch_size": 2048,
        "dtype": t.bfloat16,
    },
}

SPARSITY_PENALTIES = {
    "EleutherAI/pythia-70m-deduped": {
        "standard": [0.01, 0.05, 0.075, 0.1, 0.125, 0.15],
        "p_anneal": [0.02, 0.03, 0.035, 0.04, 0.05, 0.075],
        "gated": [0.1, 0.3, 0.5, 0.7, 0.9, 1.1],
    },
    "google/gemma-2-2b": {
        "standard": [0.025, 0.035, 0.04, 0.05, 0.06, 0.07],
        "p_anneal": [-1, -1, -1, -1, -1, -1],
        "gated": [-1, -1, -1, -1, -1, -1],
    },
}

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
        choices=["standard", "standard_new", "top_k", "gated", "p_anneal", "jump_relu"],
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
    dry_run: bool = False,
    use_wandb: bool = False,
    save_checkpoints: bool = False,
):
    # model and data parameters
    context_length = LLM_CONFIG[model_name]["context_length"]

    llm_batch_size = LLM_CONFIG[model_name]["llm_batch_size"]
    sae_batch_size = LLM_CONFIG[model_name]["sae_batch_size"]
    dtype = LLM_CONFIG[model_name]["dtype"]
    num_tokens = 50_000_000

    num_contexts_per_sae_batch = sae_batch_size // context_length
    buffer_size = num_contexts_per_sae_batch * 20

    # sae training parameters
    # random_seeds = t.arange(10).tolist()
    random_seeds = [0]
    expansion_factors = [8]

    num_sparsities = 6
    sparsity_indices = t.arange(num_sparsities).tolist()
    standard_sparsity_penalties = SPARSITY_PENALTIES[model_name]["standard"]
    p_anneal_sparsity_penalties = SPARSITY_PENALTIES[model_name]["p_anneal"]
    gated_sparsity_penalties = SPARSITY_PENALTIES[model_name]["gated"]
    ks = [20, 40, 80, 160, 320, 640]

    assert len(standard_sparsity_penalties) == num_sparsities
    assert len(p_anneal_sparsity_penalties) == num_sparsities
    assert len(gated_sparsity_penalties) == num_sparsities
    assert len(ks) == num_sparsities

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train
    warmup_steps = 1000  # Warmup period at start of training and after each resample
    resample_steps = None

    # note: learning rate is not used for topk
    learning_rates = [3e-4]

    # topk sae training parameters
    decay_start = 24000
    auxk_alpha = 1 / 32

    # p_anneal sae training parameters
    p_start = 1
    p_end = 0.2
    anneal_end = None  # steps - int(steps/10)
    sparsity_queue_length = 10
    anneal_start = 10000
    n_sparsity_updates = 10

    # jumprelu sae training parameters
    jumprelu_bandwidth = 0.001
    jumprelu_sparsity_penalty = 1.0 # per figure 9 in the paper

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

    model = LanguageModel(model_name, dispatch=True, device_map=DEVICE)
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
        if "p_anneal" in architectures:
            trainer_configs.append(
                {
                    "trainer": PAnnealTrainer,
                    "dict_class": AutoEncoder,
                    "activation_dim": activation_dim,
                    "dict_size": expansion_factor * activation_dim,
                    "lr": learning_rate,
                    "sparsity_function": "Lp^p",
                    "initial_sparsity_penalty": p_anneal_sparsity_penalties[sparsity_index],
                    "p_start": p_start,
                    "p_end": p_end,
                    "anneal_start": int(anneal_start),
                    "anneal_end": anneal_end,
                    "sparsity_queue_length": sparsity_queue_length,
                    "n_sparsity_updates": n_sparsity_updates,
                    "warmup_steps": warmup_steps,
                    "resample_steps": resample_steps,
                    "steps": steps,
                    "seed": seed,
                    "wandb_name": f"PAnnealTrainer-pythia70m-{layer}",
                    "layer": layer,
                    "lm_name": model_name,
                    "device": device,
                    "submodule_name": submodule_name,
                },
            )
        if "standard" in architectures:
            trainer_configs.append(
                {
                    "trainer": StandardTrainer,
                    "dict_class": AutoEncoder,
                    "activation_dim": activation_dim,
                    "dict_size": expansion_factor * activation_dim,
                    "lr": learning_rate,
                    "l1_penalty": standard_sparsity_penalties[sparsity_index],
                    "warmup_steps": warmup_steps,
                    "resample_steps": resample_steps,
                    "seed": seed,
                    "wandb_name": f"StandardTrainer-{model_name}-{submodule_name}",
                    "layer": layer,
                    "lm_name": model_name,
                    "device": device,
                    "submodule_name": submodule_name,
                }
            )
        if "standard_new" in architectures:
            trainer_configs.append(
                {
                    "trainer": StandardTrainer,
                    "dict_class": AutoEncoderNew,
                    "activation_dim": activation_dim,
                    "dict_size": expansion_factor * activation_dim,
                    "lr": learning_rate,
                    "l1_penalty": standard_sparsity_penalties[sparsity_index],
                    "warmup_steps": warmup_steps,
                    "resample_steps": resample_steps,
                    "seed": seed,
                    "wandb_name": f"StandardTrainerNew-{model_name}-{submodule_name}",
                    "layer": layer,
                    "lm_name": model_name,
                    "device": device,
                    "submodule_name": submodule_name,
                }
            )
        if "top_k" in architectures:
            trainer_configs.append(
                {
                    "trainer": TrainerTopK,
                    "dict_class": AutoEncoderTopK,
                    "activation_dim": activation_dim,
                    "dict_size": expansion_factor * activation_dim,
                    "k": ks[sparsity_index],
                    "auxk_alpha": auxk_alpha,  # see Appendix A.2
                    "decay_start": decay_start,  # when does the lr decay start
                    "steps": steps,  # when when does training end
                    "seed": seed,
                    "wandb_name": f"TopKTrainer-{model_name}-{submodule_name}",
                    "device": device,
                    "layer": layer,
                    "lm_name": model_name,
                    "submodule_name": submodule_name,
                }
            )
        if "gated" in architectures:
            trainer_configs.append(
                {
                    "trainer": GatedSAETrainer,
                    "dict_class": GatedAutoEncoder,
                    "activation_dim": activation_dim,
                    "dict_size": expansion_factor * activation_dim,
                    "lr": learning_rate,
                    "l1_penalty": gated_sparsity_penalties[sparsity_index],
                    "warmup_steps": warmup_steps,
                    "resample_steps": resample_steps,
                    "seed": seed,
                    "wandb_name": f"GatedSAETrainer-{model_name}-{submodule_name}",
                    "device": device,
                    "layer": layer,
                    "lm_name": model_name,
                    "submodule_name": submodule_name,
                }
            )
        if "jump_relu" in architectures:
            trainer_configs.append(
                {
                    "trainer": JumpReluTrainer,
                    "dict_class": JumpReluAutoEncoder,
                    "activation_dim": activation_dim,
                    "dict_size": expansion_factor * activation_dim,
                    "lr": learning_rate,
                    "target_l0": ks[sparsity_index],
                    "sparsity_penalty": jumprelu_sparsity_penalty,
                    "bandwidth": jumprelu_bandwidth,
                    "seed": seed,
                    "wandb_name": f"JumpReLUSAETrainer-{model_name}-{submodule_name}",
                    "device": device,
                    "layer": layer,
                    "lm_name": model_name,
                    "submodule_name": submodule_name,
                }
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

    model = LanguageModel(model_name, dispatch=True, device_map=DEVICE)
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
    for layer in args.layers:
        run_sae_training(
            model_name=args.model_name,
            layer=layer,
            save_dir=args.save_dir,
            device="cuda:0",
            architectures=args.architectures,
            dry_run=args.dry_run,
            use_wandb=args.use_wandb,
        )

    ae_paths = utils.get_nested_folders(args.save_dir)

    eval_saes(args.model_name, ae_paths, 1000, DEVICE)

    
