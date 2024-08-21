# %%
# Imports
import torch as t
from nnsight import LanguageModel
import argparse
import itertools
import gc

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainers.jump_relu import JumpReluTrainer
from dictionary_learning.utils import zst_to_generator
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder, JumpReluAutoEncoder

# %%
DEVICE = "cuda:0"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True, help="where to store sweep")
    parser.add_argument("--no_wandb_logging", action="store_true", help="omit wandb logging")
    parser.add_argument("--dry_run", action="store_true", help="dry run sweep")
    parser.add_argument("--layer", type=int, required=True, help="layer to train SAE on")
    args = parser.parse_args()
    return args


def run_sae_training(
    layer: int,
    save_dir: str,
    device: str,
    dry_run: bool = False,
    no_wandb_logging: bool = False,
):

    # model and data parameters
    model_name = "openai-community/gpt2"
    # model_name = "EleutherAI/pythia-70m-deduped"
    dataset_name = '/share/data/datasets/pile/the-eye.eu/public/AI/pile/train/00.jsonl.zst'
    context_length = 64

    buffer_size = int(1e4)
    llm_batch_size = 128  # 256 for A100 GPU, 64 for 1080ti
    sae_batch_size = 4096
    num_tokens = 200_000_000

    # sae training parameters
    # random_seeds = t.arange(10).tolist()
    random_seed = 0
    # expansion_factors = [2, 4, 8, 16, 32, 64,]# 128, 256, 512, 1024]
    expansion_factors = [16] # 128, 256, 512, 1024]
    set_linear_to_constants = [True, False]
    # set_linear_to_constants = [False]

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train
    save_steps = None
    resample_steps = None
    warmup_steps = 10000
    normalize_activations = True

    learning_rate = 7e-5
    # learning_rate = 1e-4
    l0_penalty = 1e-1

    use_wandb = not no_wandb_logging
    log_steps = 5  # Log the training on wandb
    wandb_entity = 'sae-training'
    wandb_project = 'sae-training'
    if no_wandb_logging:
        print("Not logging to wandb")
        log_steps = None

    model = LanguageModel(model_name, dispatch=True, device_map=DEVICE)
    submodule = model.transformer.h[layer]
    # submodule = model.gpt_neox.layers[layer]
    submodule_name = f"resid_post_layer_{layer}"
    io = "out"
    activation_dim = model.config.hidden_size

    generator = zst_to_generator(dataset_name)

    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=buffer_size,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        normalize_activations=normalize_activations,
        io=io,
        d_submodule=activation_dim,
        device=device,
    )


    # create the list of configs
    trainer_configs = []
    for expansion_factor, set_linear_to_constant in itertools.product(expansion_factors, set_linear_to_constants):
        trainer_configs.extend([{
            "trainer": JumpReluTrainer,
            "dict_class": JumpReluAutoEncoder,
            "activation_dim": activation_dim,
            "dict_size": expansion_factor * activation_dim,
            "lr": learning_rate,
            "l0_penalty": l0_penalty,
            "seed": random_seed,
            "warmup_steps": warmup_steps,
            "wandb_name": f"JumpReluTrainer-{model_name}-{submodule_name}-constant{set_linear_to_constant}",
            "layer": layer,
            "lm_name": model_name,
            "device": device,
            "submodule_name": submodule_name,
            "set_linear_to_constant": set_linear_to_constant,
        }])

    print(f"len trainer configs: {len(trainer_configs)}")
    save_dir = f"{save_dir}/{submodule_name}"

    if not dry_run:
        # actually run the sweep
        trainSAE(
            data=activation_buffer,
            trainer_configs=trainer_configs,
            steps=steps,
            save_steps=save_steps,
            save_dir=save_dir,
            log_steps=log_steps,
            use_wandb=use_wandb,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project,
        )


# if __name__ == "__main__":
#     args = get_args()
#     run_sae_training(
#         layer=args.layer,
#         save_dir=args.save_dir,
#         device="cuda:0",
#         dry_run=args.dry_run,
#         no_wandb_logging=args.no_wandb_logging,
#     )

if __name__ == "__main__":
    run_sae_training(
        layer=8,
        save_dir='/share/u/can/shift_eval/train_saes/trained_saes/gpt2_jumpConst_sweep0808',
        device="cuda:0",
        dry_run=False,
        no_wandb_logging=False,
    )