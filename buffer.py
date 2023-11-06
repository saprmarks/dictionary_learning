import torch as t
import zstandard as zstd
import json
import io
from nnsight import LanguageModel

"""
Implements a buffer of activations
"""

class EmptyStream(Exception):
    """
    An exception for when the data stream has been exhausted
    """
    def __init__(self):
        super().__init__()

class ActivationBuffer:
    def __init__(self, 
                 data, # generator which yields text data
                 model, # LanguageModel from which to extract activations
                 submodule, # submodule of the model from which to extract activations
                 n_ctxs=5e5, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 in_batch_size=32, # size of batches in which to process the data when adding to buffer
                 out_batch_size=4096, # size of batches in which to return activations
                 is_hf=False,
                 device='cpu'
                 ):
        
        self.activations = t.empty(0, submodule.out_features)
        self.read = t.empty(0, submodule.out_features).bool() # has the activation been read?

        self.data = data
        self.model = model # assumes model is already on device
        self.submodule = submodule
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.in_batch_size = in_batch_size
        self.out_batch_size = out_batch_size
        self.is_hf = is_hf
        self.device=device
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.n_ctxs * self.ctx_len // 2:
                try:
                    self.refresh()
                except EmptyStream: # if the data stream is exhausted, stop
                    raise StopIteration

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads))[:self.out_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]
    
    def text_batch(self):
        """
        Return a list of text
        """
        if self.is_hf:
            return [
                next(self.data)["text"] for _ in range(self.in_batch_size)
            ]
        else:
            return [
                next(self.data) for _ in range(self.in_batch_size)
            ]
    
    
    def tokenized_batch(self):
        """
        Return a batch of tokenized inputs.
        """
        texts = self.text_batch()
        return self.model.tokenizer(
            texts,
            return_tensors='pt',
            max_length=self.ctx_len,
            padding=True,
            truncation=True
        ).to(self.device)

    def refresh(self):
        """
        Refresh the buffer
        """
        # print("refreshing buffer...")

        # get rid of read activations
        self.activations = self.activations[~self.read]

        # read in new activations until buffer is full
        while len(self.activations) < self.n_ctxs * self.ctx_len:
            inputs = self.tokenized_batch()
            with self.model.generate(max_new_tokens=1, pad_token_id=self.model.tokenizer.pad_token_id) as generator:
                with generator.invoke(inputs['input_ids'], scan=False) as invoker:
                    hidden_states = self.submodule.output.save()
            # TODO once nnsight memory issue is fixed change to below:
            # with self.model.forward(inputs, scan=False) as invoker:
            #     hidden_states = self.submodule.output.save()
            attn_mask = invoker.input['attention_mask']
            tokens, attn_mask = inputs['input_ids'], inputs['attention_mask']
            self.activations = t.cat(
                [self.activations, hidden_states.value[attn_mask == 1].to('cpu')], # activations over non-padding tokens
                dim=0
            )
        
        self.read = t.zeros(len(self.activations)).bool()
        # print('buffer refreshed...')

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()
