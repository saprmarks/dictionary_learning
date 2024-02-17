This is a repository for doing dictionary learning via sparse autoencoders on neural network activations. It was developed by Samuel Marks and Aaron Mueller. 

For accessing, saving, and intervening on NN activations, we use the [`nnsight`](http://nnsight.net/) package; as of December 2023, nnsight is under active development and may undergo breaking changes. That said, `nnsight` is easy to use and quick to learn; if you plan to modify this repo, then we recommend going through the main `nnsight` demo [here](https://nnsight.net/notebooks/tutorials/walkthrough/).

Some dictionaries trained using this repository (and asociated training checkpoints) can be accessed at [https://baulab.us/u/smarks/autoencoders/](https://baulab.us/u/smarks/autoencoders/). See below for more information about these dictionaries.

# Set-up

Navigate to the to the location where you would like to clone this repo, clone and enter the repo, and install the requirements.
```bash
git clone https://github.com/saprmarks/dictionary_learning
cd dictionary_learning
pip install -r requirements.txt
```

To use `dictionary_learning`, include it as a subdirectory in some project's directory and import it; see the examples below.

# Using trained dictionaries

To use a dictionary, just import the dictionary class (currently only autoencoders are supported), initialize an `AutoEncoder`, and load a saved state_dict.
```python
from dictionary_learning import AutoEncoder
import torch

activation_dim = 512 # dimension of the NN's activations to be autoencoded
dictionary_size = 16 * activation_dim # number of features in the dictionary
ae = AutoEncoder(activation_dim, dictionary_size)
ae.load_state_dict(torch.load("path/to/dictionary/weights"))

# get NN activations using your preferred method: hooks, transformer_lens, nnsight, etc. ...
# for now we'll just use random activations
activations = torch.randn(64, activation_dim)
features = ae.encode(activations) # get features from activations
reconstructed_activations = ae.decode(features)

# if you want to use both the features and the reconstruction, you can get both at once
reconstructed_activations, features = ae(activations, output_features=True)
```

Dictionaries have `encode`, `decode`, and `forward` methods -- see `dictionary.py`.

# Training your own dictionaries

To train your own dictionaries, you'll need to understand a bit about our infrastructure.

One key object is the `ActivationBuffer`, defined in `buffer.py`. Following [Neel Nanda's appraoch](https://www.lesswrong.com/posts/fKuugaxt2XLTkASkk/open-source-replication-and-commentary-on-anthropic-s), `ActivationBuffer`s maintain a buffer of NN activations, which it outputs in batches.

An `ActivationBuffer` is initialized from an `nnsight` `LanguageModel` object, a submodule (e.g. an MLP), and a generator which yields strings (the text data). It processes a large number of strings, up to some capacity, and saves the submodule's activations. You sample batches from it, and when it is half-depleted, it refreshes itself with new text data.

Here's an example for training a dictionary; in it we load a language model as an `nnsight` `LanguageModel` (this will work for any Huggingface model), specify a submodule, create an `ActivationBuffer`, and then train an autoencoder with `trainSAE`.
```python
from nnsight import LanguageModel
from dictionary_learning import ActivationBuffer
from dictionary_learning.training import trainSAE

model = LanguageModel(
    'EleutherAI/pythia-70m-deduped', # this can be any Huggingface model
    device_map = 'cuda:0'
)
submodule = model.gpt_neox.layers[1].mlp # layer 1 MLP
activation_dim = 512 # output dimension of the MLP
dictionary_size = 16 * activation_dim

# data much be an iterator that outputs strings
data = iter([
    'This is some example data',
    'In real life, for training a dictionary',
    'you would need much more data than this'
])
buffer = ActivationBuffer(
    data,
    model,
    submodule,
    out_feats=activation_dim, # output dimension of the model component
    n_ctxs=3e4, # you can set this higher or lower dependong on your available memory
    device='cuda:0' # doesn't have to be the same device that you train your autoencoder on
) # buffer will return batches of tensors of dimension = submodule's output dimension

# train the sparse autoencoder (SAE)
ae = trainSAE(
    buffer,
    activation_dim,
    dictionary_size,
    lr=3e-4,
    sparsity_penalty=1e-3,
    device='cuda:0'
)
```
Some technical notes our training infrastructure and supported features:
* Training uses the `ConstrainedAdam` optimizer defined in `training.py`. This is a variant of Adam which supports constraining the `AutoEncoder`'s decoder weights to be norm 1.
* Neuron resampling: if a `resample_steps` argument is passed to `trainSAE`, then dead neurons will periodically be resampled according to the procedure specified [here](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-resampling).
* Ghost grads: if a `ghost_threshold` argument is passed to `trainSAE`, then [ghost grads](https://transformer-circuits.pub/2024/jan-update/index.html#dict-learning-resampling) is used. Neurons which haven't fired in more than `ghost_threshold` steps are treated as dead for purposes of ghost grads.
* Learning rate warmup: if a `warmup_steps` argument is passed to `trainSAE`, then a linear LR warmup is used at the start of training and, if doing neuron resampling, also after every time neurons are resampled.

If `submodule` is a model component where the activations are tuples (e.g. this is common when working with residual stream activations), then the buffer yields the first coordinate of the tuple.

# Downloading our open-source dictionaries

To download our open-source dictionaries and associated training checkpoints, navigate to the directory you would like to save the dictionaries in, and then:
```bash
wget -r --no-parent https://baulab.us/u/smarks/autoencoders/
```

Currently, the main thing to look for is the dictionaries in our `5_32768` set; this set has dictionaries for MLP outputs, attention outputs, and residual streams in all layers of EleutherAI's Pythia-70m-deduped model. These dictionaries were trained on 2B tokens from the pile with neuron resampling every 250M tokens.

Let's explain the directory structure by example. The `autoencoders/pythia-70m-deduped/mlp_out_layer1/5_32768` directory corresponds to the layer 1 MLP dictionary from `5_32768` set. This directory contains:
* `ae.pt`: the `state_dict` of the fully trained dictionary
* `config.json`: a json file which specifies the hyperparameters used to train the dictionary
* `checkpoints/`: a directory containing training checkpoints of the form `ae_step.pt`.

There are also MLP three sets of MLP output dictionaries, all for Pythia-70m-deduped: `0_8192`, `1_32768`, and `2_32768` (the number after the underscore indicates the hidden dimension of the autoencoder). For more information about the `0_8192` and `1_32768` sets, see [here](https://www.lesswrong.com/posts/AaoWLcmpY3LKvtdyq/some-open-source-dictionaries-and-dictionary-learning). The `2_32768` set is shrouded in mystery and man may never know its true nature.

## Statistics for our dictionaries

We'll report the following statistics for our `5_32768` set. These were measured using the code in `evaluation.py`.
* **MSE loss**: average squared L2 distance between an activation and the autoencoder's reconstruction of it
* **L1 loss**: a measure of the autoencoder's sparsity
* **L0**: average number of features active above a random token
* **Percentage of neurons alive**: fraction of the dictionary features which are active on at least one token out of 8192 random tokens
* **CE diff**: difference between the usual cross-entropy loss of the model for next token prediction and the cross entropy when replacing activations with our dictionary's reconstruction
* **Percentage of CE loss recovered**: when replacing the activation with the dictionary's reconstruction, the percentage of the model's cross-entropy loss on next token prediction that is recovered (relative to the baseline of zero ablating the activation)



### MLP output dictionaries

| Layer         | MSE Loss | % Variance Explained | L1 | L0   | % Alive | CE Diff | % CE Recovered |
|---------------|----------|--------------------|---------------|------|---------------|---------|-------------------|
| 0 | 0.0018   | 97               | 6.3           | 9.4  | 37          | 0.050   | 99                |
| 1 | 0.0090   | 78               | 4.9           | 22.9 | 48          | 0.080   | 87                |
| 2 | 0.015    | 98               | 7.8           | 23.8 | 31          | 0.12    | 77                |
| 3 | 0.042    | 75               | 10.8          | 44.1 | 19          | 0.19    | 74                |
| 4 | 0.050    | 86               | 12            | 27.2 | 22          | 0.21    | 78                |
| 5 | 0.093    | 91               | 21            | 20.7 | 6.6         | 0.30    | 90                |

### Residual stream dictionaries
NOTE: the layer indices here are, confusingly, offset by 1. So the layer 0 dictionaries is not for the embeddings -- it's for the residual stream at the *end* of layer 0, i.e. what is normally called the layer 1 residual stream. Sorry about the confusion, hopefully this won't happen in future dictionary releases.

| Layer           | MSE Loss | % Variance Explained | L1 | L0   | % Alive | CE Diff | % CE Recovered |
|-----------------|----------|--------------------|---------------|------|---------------|---------|-------------------|
| 0 | 0.012    | 85               | 7.7           | 17   | 27          | 0.30    | 94                |
| 1 | 0.031    | 76               | 8.8           | 15.9 | 26          | 0.54    | 89                |
| 2 | 0.064    | 93               | 15            | 34.8 | 24          | 1.4     | 76                |
| 3 | 0.066    | 93               | 15            | 22.6 | 20          | 1.1     | 88                |
| 4 | 0.098    | 81               | 14            | 17.7 | 17          | 0.89    | 83                |
| 5 | 0.21     | 82               | 22            | 15.2 | 9.3         | 1.4     | 73                |

### Attention output dictionaries

| Layer          | MSE Loss | % Variance Explained | L1 | L0   | % Alive | CE Diff | % CE Recovered |
|----------------|----------|--------------------|---------------|------|---------------|---------|-------------------|
| 0 | 0.0042   | 85               | 5.2           | 35   | 17          | 0.055   | 96                |
| 1 | 0.0076   | 76               | 4.9           | 28.4 | 15          | 0.068   | 85                |
| 2 | 0.022    | 75               | 10            | 59.9 | 10          | 0.19    | 76                |
| 3 | 0.012    | 78               | 6.5           | 34.2 | 10          | 0.10    | 83                |
| 4 | 0.0075   | 65               | 3.7           | 21.3 | 14          | 0.029   | 89                |
| 5 | 0.014    | 76               | 5.3           | 17.7 | 7.6         | 0.060   | 82                |


# Extra functionality supported by this repo

**Note:** these features are likely to be depricated in future releases.

We've included support for some experimental features. We briefly investigated them as an alternative approaches to training dictionaries.

* **MLP stretchers.** Based on the perspective that one may be able to identify features with "[neurons in a sufficiently large model](https://transformer-circuits.pub/2022/toy_model/index.html)," we experimented with training "autoencoders" to, given as input an MLP *input* activation $x$, output not $x$ but $MLP(x)$ (the same output as the MLP). For instance, given an MLP which maps a 512-dimensional input $x$ to a 1024-dimensional hidden state $h$ and then a 512-dimensional output $y$, we train a dictionary $A$ with hidden dimension 16384 = 16 x 1024 so that $A(x)$ is close to $y$ (and, as usual, so that the hidden state of the dictionary is sparse).
    * The resulting dictionaries seemed decent, but we decided not to pursue the idea further.
    * To use this functionality, set the `io` parameter of an activaiton buffer to `'in_to_out'` (default is `'out'`).
    * h/t to Max Li for this suggestion.
* **Replacing L1 loss with entropy**. Based on the ideas in this [post](https://transformer-circuits.pub/2023/may-update/index.html#simple-factorization), we experimented with using entropy to regularize a dictionary's hidden state instead of L1 loss. This seemed to cause the features to split into dead features (which never fired) and very high-frequency features which fired on nearly every input, which was not the desired behavior. But plausibly there is a way to make this work better.


