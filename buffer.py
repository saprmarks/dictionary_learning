import torch as t
from nnsight import LanguageModel
from .config import DEBUG

if DEBUG:
    tracer_kwargs = {'scan' : True, 'validate' : True}
else:
    tracer_kwargs = {'scan' : False, 'validate' : False}

"""
Implements a buffer of activations
outputs activations of shape (n_ctxs, ctx_len, submodule_input(output)_dim)
"""

class ActivationBuffer:
    def __init__(self, 
                 data, # generator which yields text data
                 model : LanguageModel, # LanguageModel from which to extract activations
                 submodule, # submodule of the model from which to extract activations
                 submodule_input_dim=None,
                 submodule_output_dim=None,
                 io='out', # whether to extract input or output activations
                 n_ctxs=3e4, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 load_buffer_batch_size=512, # size of batches in which to process the data when adding to buffer
                 return_act_batch_size=8192, # size of batches in which to return activations
                 device='cpu' # device on which to store the activations
                 ):
        
        if io == 'in':
            if submodule_input_dim is None:
                try:
                    submodule_input_dim = submodule.in_features
                except:
                    raise ValueError("submodule_input_dim cannot be inferred and must be specified directly")
            self.activations = t.empty(0, submodule_input_dim, device=device)

        elif io == 'out':
            if submodule_output_dim is None:
                try:
                    submodule_output_dim = submodule.out_features
                except:
                    raise ValueError("submodule_output_dim cannot be inferred and must be specified directly")
            self.activations = t.empty(0, submodule_output_dim, device=device)
        else:
            raise ValueError("io must be either 'in' or 'out'")
        self.read = t.zeros(0).bool()

        self.data = data
        self.model = model
        self.submodule = submodule
        self.io = io
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.load_buffer_batch_size = load_buffer_batch_size
        self.return_act_batch_size = return_act_batch_size
        self.device = device
    
    def __iter__(self):
        return self

    def __next__(self):
        """
        Return a batch of activations
        """
        with t.no_grad():
            # if buffer is less than half full, refresh
            if (~self.read).sum() < self.n_ctxs * self.ctx_len // 2:
                self.refresh()

            # return a batch
            unreads = (~self.read).nonzero().squeeze()
            idxs = unreads[t.randperm(len(unreads), device=unreads.device)[:self.return_act_batch_size]]
            self.read[idxs] = True
            return self.activations[idxs]
    
    def text_batch(self, batch_size=None):
        """
        Return a list of text
        """
        if batch_size is None:
            batch_size = self.load_buffer_batch_size
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
        self.activations = self.activations[~self.read]

        while len(self.activations) < self.n_ctxs * self.ctx_len:
            
            with t.no_grad():
                with self.model.trace(self.text_batch(), **tracer_kwargs, invoker_args={'truncation': True, 'max_length': self.ctx_len}):
                    if self.io == 'in':
                        hidden_states = self.submodule.input[0].save()
                    else:
                        hidden_states = self.submodule.output.save()
                    input = self.model.input.save()
            attn_mask = input.value[1]['attention_mask']
            hidden_states = hidden_states.value
            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
            hidden_states = hidden_states[attn_mask != 0]
            self.activations = t.cat([self.activations, hidden_states.to(self.device)], dim=0)
            self.read = t.zeros(len(self.activations), dtype=t.bool, device=self.device)

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()