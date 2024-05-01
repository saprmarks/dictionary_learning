from nnsight import LanguageModel

from .buffer import ActivationBuffer
from .training import trainSAE

EXPANSION_FACTOR = 8
DEVICE = "mps"

model = LanguageModel(
    "EleutherAI/pythia-70m-deduped", device_map=DEVICE  # this can be any Huggingface model
)

submodule = model.gpt_neox.layers[1].mlp  # layer 1 MLP
activation_dim = 512  # dimension of the activations in pythia-70m
dictionary_size = EXPANSION_FACTOR * activation_dim

# data much be an iterator that outputs strings
data = iter(
    [
        "This is some example data",
        "In real life, for training a dictionary",
        "you would need much more data than this",
    ]
)

buffer = ActivationBuffer(
    data,
    model,
    submodule,
    d_submodule=activation_dim,  # output dimension of the model component
    n_ctxs=3e4,  # you can set this higher or lower dependong on your available memory
    device=DEVICE,  # doesn't have to be the same device that you train your autoencoder on
)  # buffer will return batches of tensors of dimension = submodule's output dimension

# train the sparse autoencoder (SAE)
ae = trainSAE(
    buffer,
    activation_dim,
    dictionary_size,
    lr=3e-4,
    sparsity_penalty=1e-3,
    device=DEVICE,
    use_gated_sae=True,
)

print("Done!")
