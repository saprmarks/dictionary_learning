import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import random

from dictionary_learning.training import trainSAE
from dictionary_learning.trainers.standard import StandardTrainer
from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK
from dictionary_learning.utils import (
    hf_dataset_to_generator,
    get_nested_folders,
    load_dictionary,
)

# from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.pytorch_buffer import ActivationBuffer
from dictionary_learning.dictionary import (
    AutoEncoder,
    GatedAutoEncoder,
    AutoEncoderNew,
    JumpReluAutoEncoder,
)
from dictionary_learning.evaluation import evaluate

EXPECTED_RESULTS = {
    "AutoEncoderTopK": {
        "l2_loss": 4.358876752853393,
        "l1_loss": 50.90618553161621,
        "l0": 40.0,
        "frac_variance_explained": 0.9577824175357819,
        "cossim": 0.9476200461387634,
        "l2_ratio": 0.9476299166679383,
        "relative_reconstruction_bias": 0.9996505916118622,
        "frac_alive": 1.0,
    },
    "AutoEncoder": {
        "l2_loss": 6.8308186531066895,
        "l1_loss": 19.398421669006346,
        "l0": 37.4469970703125,
        "frac_variance_explained": 0.9003101229667664,
        "cossim": 0.8782103300094605,
        "l2_ratio": 0.7444103538990021,
        "relative_reconstruction_bias": 0.960041344165802,
        "frac_alive": 0.9970703125,
    },
}

DEVICE = "cuda:0"
SAVE_DIR = "./test_data"
MODEL_NAME = "EleutherAI/pythia-70m-deduped"
RANDOM_SEED = 42
LAYER = 3
DATASET_NAME = "monology/pile-uncopyrighted"

EVAL_TOLERANCE_PERCENT = 0.005


def test_sae_training():
    """End to end test for training an SAE. Takes ~2 minutes on an RTX 3090.
    This isn't a nice suite of unit tests, but it's better than nothing.
    I have observed that results can slightly vary with library versions. For full determinism,
    use pytorch 2.5.1 and nnsight 0.3.7.
    Unfortunately an RTX 3090 is also required for full determinism. On an H100 the results are off by ~0.3%, meaning this test will
    not be within the EVAL_TOLERANCE."""

    random.seed(RANDOM_SEED)
    t.manual_seed(RANDOM_SEED)

    # model = LanguageModel(MODEL_NAME, dispatch=True, device_map=DEVICE)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=t.float32
    ).to(DEVICE)

    context_length = 128
    llm_batch_size = 512  # Fits on a 24GB GPU
    sae_batch_size = 8192
    num_contexts_per_sae_batch = sae_batch_size // context_length

    num_inputs_in_buffer = num_contexts_per_sae_batch * 20

    num_tokens = 10_000_000

    # sae training parameters
    k = 40
    sparsity_penalty = 2.0
    expansion_factor = 8

    steps = int(num_tokens / sae_batch_size)  # Total number of batches to train
    save_steps = None
    warmup_steps = 1000  # Warmup period at start of training and after each resample
    resample_steps = None

    # standard sae training parameters
    learning_rate = 3e-4

    # topk sae training parameters
    decay_start = None
    auxk_alpha = 1 / 32

    submodule = model.gpt_neox.layers[LAYER]
    submodule_name = f"resid_post_layer_{LAYER}"
    io = "out"
    activation_dim = model.config.hidden_size

    generator = hf_dataset_to_generator(DATASET_NAME)

    activation_buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=num_inputs_in_buffer,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io=io,
        d_submodule=activation_dim,
        device=DEVICE,
    )

    # create the list of configs
    trainer_configs = []
    trainer_configs.extend(
        [
            {
                "trainer": TopKTrainer,
                "dict_class": AutoEncoderTopK,
                "lr": None,
                "activation_dim": activation_dim,
                "dict_size": expansion_factor * activation_dim,
                "k": k,
                "auxk_alpha": auxk_alpha,  # see Appendix A.2
                "warmup_steps": 0,
                "decay_start": decay_start,  # when does the lr decay start
                "steps": steps,  # when when does training end
                "seed": RANDOM_SEED,
                "wandb_name": f"TopKTrainer-{MODEL_NAME}-{submodule_name}",
                "device": DEVICE,
                "layer": LAYER,
                "lm_name": MODEL_NAME,
                "submodule_name": submodule_name,
            },
        ]
    )
    trainer_configs.extend(
        [
            {
                "trainer": StandardTrainer,
                "dict_class": AutoEncoder,
                "activation_dim": activation_dim,
                "dict_size": expansion_factor * activation_dim,
                "lr": learning_rate,
                "l1_penalty": sparsity_penalty,
                "warmup_steps": warmup_steps,
                "sparsity_warmup_steps": None,
                "decay_start": decay_start,
                "steps": steps,
                "resample_steps": resample_steps,
                "seed": RANDOM_SEED,
                "wandb_name": f"StandardTrainer-{MODEL_NAME}-{submodule_name}",
                "layer": LAYER,
                "lm_name": MODEL_NAME,
                "device": DEVICE,
                "submodule_name": submodule_name,
            },
        ]
    )

    print(f"len trainer configs: {len(trainer_configs)}")
    output_dir = f"{SAVE_DIR}/{submodule_name}"

    trainSAE(
        data=activation_buffer,
        trainer_configs=trainer_configs,
        steps=steps,
        save_steps=save_steps,
        save_dir=output_dir,
    )

    folders = get_nested_folders(output_dir)

    assert len(folders) == 2

    for folder in folders:
        dictionary, config = load_dictionary(folder, DEVICE)

        assert dictionary is not None
        assert config is not None


def test_evaluation():
    random.seed(RANDOM_SEED)
    t.manual_seed(RANDOM_SEED)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, device_map="auto", torch_dtype=t.float32
    ).to(DEVICE)
    ae_paths = get_nested_folders(SAVE_DIR)

    context_length = 128
    llm_batch_size = 100
    sae_batch_size = 4096
    n_batches = 10
    buffer_size = 256
    io = "out"

    generator = hf_dataset_to_generator(DATASET_NAME)
    submodule = model.gpt_neox.layers[LAYER]

    input_strings = []
    for i, example in enumerate(generator):
        input_strings.append(example)
        if i > buffer_size * n_batches:
            break

    for ae_path in ae_paths:
        dictionary, config = load_dictionary(ae_path, DEVICE)
        dictionary = dictionary.to(dtype=model.dtype)

        activation_dim = config["trainer"]["activation_dim"]
        context_length = config["buffer"]["ctx_len"]

        activation_buffer_data = iter(input_strings)

        activation_buffer = ActivationBuffer(
            activation_buffer_data,
            model,
            submodule,
            n_ctxs=buffer_size,
            ctx_len=context_length,
            refresh_batch_size=llm_batch_size,
            out_batch_size=sae_batch_size,
            io=io,
            d_submodule=activation_dim,
            device=DEVICE,
        )

        eval_results = evaluate(
            dictionary,
            activation_buffer,
            context_length,
            llm_batch_size,
            io=io,
            device=DEVICE,
            n_batches=n_batches,
        )

        print(eval_results)

        dict_class = config["trainer"]["dict_class"]
        expected_results = EXPECTED_RESULTS[dict_class]

        max_diff = 0
        max_diff_percent = 0
        for key, value in expected_results.items():
            diff = abs(eval_results[key] - value)
            max_diff = max(max_diff, diff)
            max_diff_percent = max(max_diff_percent, diff / value)

        print(f"Max diff: {max_diff}, max diff %: {max_diff_percent}")
        assert max_diff_percent < EVAL_TOLERANCE_PERCENT
