import torch as t
import zstandard as zstd
import json
import io
from nnsight import LanguageModel

"""
Implements a buffer of activations
"""

class ActivationBuffer:
    def __init__(self, 
                 data, # generator which yields text data
                 model, # LanguageModel from which to extract activations
                 submodules, # submodule of the model from which to extract activations
                 in_feats=None,
                 out_feats=None, 
                 io='out', # can be 'in', 'out', or 'in_to_out'
                 n_ctxs=3e4, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 in_batch_size=512, # size of batches in which to process the data when adding to buffer
                 out_batch_size=8192, # size of batches in which to return activations
                 device='cpu' # device on which to store the activations
                 ):
        
        # dictionary of activations
        self.activations = {}
        for submodule in submodules:
            if io == 'in':
                if in_feats is None:
                    try:
                        in_feats = submodule.in_features
                    except:
                        raise ValueError("in_feats cannot be inferred and must be specified directly")
                self.activations[submodule] = t.empty(0, in_feats, device=device)

            elif io == 'out':
                if out_feats is None:
                    try:
                        out_feats = submodule.out_features
                    except:
                        raise ValueError("out_feats cannot be inferred and must be specified directly")
                self.activations[submodule] = t.empty(0, out_feats, device=device)
            elif io == 'in_to_out':
                raise ValueError("Support for in_to_out is depricated")
        self.read = t.zeros(0, dtype=t.bool, device=device)
        self._n_activations = 0 # for tracking how many activations (read or unread) are currently in the buffer

        self.data = data
        self.model = model # assumes nnsight model is already on the device
        self.submodules = submodules
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.in_batch_size = in_batch_size
        self.out_batch_size = out_batch_size
        self.device = device
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        # if buffer is less than half full, refresh
        if (~self.read).sum() < self.n_ctxs * self.ctx_len // 2:
            self.refresh()

        # return a batch
        unreads = (~self.read).nonzero().squeeze()
        idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.out_batch_size]]
        self.read[idxs] = True
        return {
            submodule : activations[idxs] for submodule, activations in self.activations.items()
        }
    
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.in_batch_size
        try:
            return [
                next(self.data) for _ in range(batch_size)
            ]
        except StopIteration:
            raise StopIteration("End of data stream reached")
    
    def tokenized_batch(self, batch_size=None):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch(batch_size=batch_size)
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True
        )

    def refresh(self):
        for submodule, activations in self.activations.items():
            self.activations[submodule] = activations[~self.read]
        self._n_activations = (~self.read).sum().item()

        while self._n_activations < self.n_ctxs * self.ctx_len:
                
                with self.model.invoke(self.text_batch(), truncation=True, max_length=self.ctx_len) as invoker:
                    hidden_states = {}
                    for submodule in self.submodules:
                        if self.io == 'in':
                            x = submodule.input
                        else:
                            x = submodule.output
                        if (type(x.shape) == tuple):
                            x = x[0]
                        hidden_states[submodule] = x.save()

                attn_mask = invoker.input['attention_mask']
                
                self._n_activations += (attn_mask != 0).sum().item()     

                for submodule, activations in self.activations.items():
                    self.activations[submodule] = t.cat((
                        activations,
                        hidden_states[submodule].value[attn_mask != 0].to(activations.device)),
                        dim=0
                    )
                    assert len(self.activations[submodule]) == self._n_activations

        self.read = t.zeros(self._n_activations, dtype=t.bool, device=self.device)

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()
