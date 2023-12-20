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
from datasets import load_dataset
from einops import rearrange
import torch as t

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
        input_tokens,
        add_residual=False, # whether to compensate for dictionary reconstruction error by adding residual
        k=10,
):
    """
    Effect of ablating the feature on top k predictions for next token.
    """
    # clean run
    with model.invoke(input_tokens) as invoker:
        if not add_residual: # run hidden state through autoencoder
            if type(submodule.output.shape) == tuple:
                submodule.output[0][:] = dictionary(submodule.output[0])
            else:
                submodule.output = dictionary(submodule.output)
    clean_probs = invoker.output.logits[0, -1, :].softmax(dim=-1)

    # ablated run
    with model.invoke(input_tokens) as invoker:
        if type(submodule.output.shape) == tuple:
            x = submodule.output[0]
        else:
            x = submodule.output

        f = dictionary.encode(x)   
        f[0, -1, feature] = 0
        if not add_residual:
            x = dictionary.decode(f)
        else:
            residual = dictionary(x) - x
            x = dictionary.decode(f) - residual
        
        if type(submodule.output.shape) == tuple:
            submodule.output[0][:] = x
        else:
            submodule.output = x
    
    ablated_probs = invoker.output.logits[0, -1, :].softmax(dim=-1)
    diff = clean_probs - ablated_probs

    top_probs, top_tokens = diff.topk(k)
    return top_tokens, top_probs
