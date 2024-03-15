import zstandard as zstd
import json
import os
import io
import random
from tqdm import tqdm
from nnsight import LanguageModel
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.dictionary import AutoEncoder
from dictionary_learning.training import trainSAE
from collections import defaultdict
from circuitsvis.activations import text_neuron_activations
from circuitsvis.topk_tokens import topk_tokens
from datasets import load_dataset
from einops import rearrange
import torch as t
from collections import namedtuple
import umap
import pandas as pd
import plotly.express as px

def list_decode(model, x):
    if isinstance(x, int):
        return model.tokenizer.decode(x)
    else:
        return [list_decode(model, y) for y in x]
    

def random_feature(model, submodule, autoencoder, buffer,
                   num_examples=10):
    inputs = buffer.tokenized_batch()
    with model.generate(max_new_tokens=1, pad_token_id=model.tokenizer.pad_token_id) as generator:
        with generator.invoke(inputs['input_ids'], scan=False) as invoker:
            hidden_states = submodule.output.save()
    dictionary_activations = autoencoder.encode(hidden_states.value)
    num_features = dictionary_activations.shape[2]
    feat_idx = random.randint(0, num_features-1)
    
    flattened_acts = rearrange(dictionary_activations, 'b n d -> (b n) d')
    acts = dictionary_activations[:, :, feat_idx].cpu()
    flattened_acts = rearrange(acts, 'b l -> (b l)')
    top_indices = t.argsort(flattened_acts, dim=0, descending=True)[:num_examples]
    batch_indices = top_indices // acts.shape[1]
    token_indices = top_indices % acts.shape[1]

    tokens = [
        inputs['input_ids'][batch_idx, :token_idx+1].tolist() for batch_idx, token_idx in zip(batch_indices, token_indices)
    ]
    tokens = list_decode(model, tokens)
    activations = [
        acts[batch_idx, :token_id+1, None, None] for batch_idx, token_id in zip(batch_indices, token_indices)
    ]

    return (feat_idx, tokens, activations)

def feature_effect(
        model,
        submodule,
        dictionary,
        feature,
        inputs,
        add_residual=True, # whether to compensate for dictionary reconstruction error by adding residual
        k=10,
        largest=True,
):
    """
    Effect of ablating the feature on top k predictions for next token.
    """
    # clean run
    with model.invoke(inputs) as invoker:
        if dictionary is None:
            pass
        elif not add_residual: # run hidden state through autoencoder
            if type(submodule.output.shape) == tuple:
                submodule.output[0][:] = dictionary(submodule.output[0])
            else:
                submodule.output = dictionary(submodule.output)
    clean_logits = invoker.output.logits[:, -1, :]
    clean_logprobs = t.nn.functional.log_softmax(clean_logits, dim=-1)

    # ablated run
    with model.invoke(inputs) as invoker:
        if dictionary is None:
            if type(submodule.output.shape) == tuple:
                submodule.output[0][:, -1, feature] = 0
            else:
                submodule.output[:, -1, feature] = 0
        else:
            x = submodule.output
            if type(x.shape) == tuple:
                x = x[0]
            x_hat, f = dictionary(x, output_features=True)
            residual = x - x_hat
            
            f[:, -1, feature] = 0
            if add_residual:
                x_hat = dictionary.decode(f) + residual
            else:
                x_hat = dictionary.decode(f)
            
            if type(submodule.output.shape) == tuple:
                submodule.output[0][:] = x_hat
            else:
                submodule.output = x_hat
    
    ablated_logits = invoker.output.logits[:, -1, :]
    ablated_logprobs = t.nn.functional.log_softmax(ablated_logits, dim=-1)
    logit_diff = clean_logits - ablated_logits
    logprob_diff = clean_logprobs - ablated_logprobs

    top_logits, top_logit_tokens = t.topk(logit_diff.mean(dim=0), k=k, largest=largest)
    top_logprobs, top_logprob_tokens = t.topk(logprob_diff.mean(dim=0), k=k, largest=largest)

    return top_logits, top_logit_tokens, top_logprobs, top_logprob_tokens


def examine_dimension(model, submodule, buffer, dictionary=None, max_length=128, n_inputs=512,
                      dim_idx=None, k=30):
    
    def _list_decode(x):
        if isinstance(x, int):
            return model.tokenizer.decode(x)
        else:
            return [_list_decode(y) for y in x]
    
    if dim_idx is None:
        dim_idx = random.randint(0, activations.shape[-1]-1)

    inputs = buffer.text_batch(batch_size=n_inputs)
    with model.invoke(inputs, max_length=max_length, truncation=True) as invoker:
        activations = submodule.output
        if type(activations.shape) == tuple:
            activations = activations[0]
        if dictionary is not None:
            activations = dictionary.encode(activations)
        activations = activations[:,:, dim_idx].save()
    activations = activations.value

    # get top k tokens by mean activation
    tokens = invoker.input['input_ids']
    token_mean_acts = {}
    for ctx in tokens:
        for tok in ctx:
            if tok.item() in token_mean_acts:
                continue
            idxs = (tokens == tok).nonzero(as_tuple=True)
            token_mean_acts[tok.item()] = activations[idxs].mean().item()
    sorted_tokens = sorted(token_mean_acts.items(), key=lambda x: x[1], reverse=True)
    top_tokens = sorted_tokens[:k]
    top_tokens = [(model.tokenizer.decode(tok), act) for tok, act in top_tokens]
    bottom_tokens = sorted_tokens[-k:]
    bottom_tokens = [(model.tokenizer.decode(tok), act) for tok, act in bottom_tokens]

    flattened_acts = rearrange(activations, 'b n -> (b n)')
    topk_indices = t.argsort(flattened_acts, dim=0, descending=True)[:k]
    batch_indices = topk_indices // activations.shape[1]
    token_indices = topk_indices % activations.shape[1]
    tokens = [
        tokens[batch_idx, :token_idx+1].tolist() for batch_idx, token_idx in zip(batch_indices, token_indices)
    ]
    activations = [
        activations[batch_idx, :token_id+1, None, None] for batch_idx, token_id in zip(batch_indices, token_indices)
    ]
    decoded_tokens = _list_decode(tokens)
    top_contexts = text_neuron_activations(decoded_tokens, activations)

    top_logits, top_logit_tokens, top_logprobs, top_logprob_tokens = feature_effect(
        model,
        submodule,
        dictionary,
        dim_idx,
        tokens,
        k=k
    )
    top_affected_logits = [(model.tokenizer.decode(tok), prob.item()) for tok, prob in zip(top_logit_tokens, top_logits)]
    top_affected_logprobs = [(model.tokenizer.decode(tok), prob.item()) for tok, prob in zip(top_logprob_tokens, top_logprobs)]

    return namedtuple(
        'featureProfile',
        ['top_contexts', 'top_tokens', 'bottom_tokens', 'top_affected_logits', 'top_affected_logprobs']
    )(
        top_contexts, top_tokens, bottom_tokens, top_affected_logits, top_affected_logprobs
    )

def feature_umap(
        dictionary,
        weight='decoder', # 'encoder' or 'decoder'
        # UMAP parameters
        n_neighbors=15,
        metric='cosine',
        min_dist=0.05,
        n_components=2, # dimension of the UMAP embedding
        feat_idxs=None, # if not none, indicate the feature with a red dot
):
    """
    Fit a UMAP embedding of the dictionary features and return a plotly plot of the result."""
    if weight == 'encoder':
        df = pd.DataFrame(dictionary.encoder.weight.cpu().detach().numpy())
    else:
        df = pd.DataFrame(dictionary.decoder.weight.T.cpu().detach().numpy())
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        metric=metric,
        min_dist=min_dist,
        n_components=n_components,
    )
    embedding = reducer.fit_transform(df)
    if feat_idxs is None:
        colors = None
    if isinstance(feat_idxs, int):
        feat_idxs = [feat_idxs]
    else:
        colors = ['blue' if i not in feat_idxs else 'red' for i in range(embedding.shape[0])]
    if n_components == 2:
        return px.scatter(x=embedding[:, 0], y=embedding[:, 1], hover_name=df.index, color=colors)
    if n_components == 3:
        return px.scatter_3d(x=embedding[:, 0], y=embedding[:, 1], z=embedding[:, 2], hover_name=df.index, color=colors)
    raise ValueError("n_components must be 2 or 3")