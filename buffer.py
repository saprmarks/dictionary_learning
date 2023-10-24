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
                 data, # path to a .zstd file
                 model, # LanguageModel from which to extract activations
                 submodule, # submodule of the model from which to extract activations
                 n_ctxs=5e5, # approximate number of contexts to store in the buffer
                 ctx_len=128, # length of each context
                 in_batch_size=16, # size of batches in which to process the data when adding to buffer
                 out_batch_size=4096, # size of batches in which to return activations
                 device='cpu'
                 ):
        
        self.activations = t.empty(0, submodule.out_features)
        self.read = t.empty(0, submodule.out_features).bool() # has the activation been read?

        # load the datastream
        compressed_file = open(data, 'rb')
        dctx = zstd.ZstdDecompressor()
        reader = dctx.stream_reader(compressed_file)
        self.text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        
        self.model = model # assumes model is already on device
        self.submodule = submodule
        self.n_ctxs = n_ctxs
        self.ctx_len = ctx_len
        self.in_batch_size = in_batch_size
        self.out_batch_size = out_batch_size
        self.device=device

    def get_batch(self):
        """
        Get a batch of activations
        """
        # if buffer is less than half full, refresh
        if (~self.read).sum() < self.n_ctxs * self.ctx_len // 2:
            self.refresh()
        
        # otherwise, return a batch
        unreads = (~self.read).nonzero().squeeze()
        idxs = unreads[t.randperm(len(unreads))[:self.out_batch_size]]
        self.read[idxs] = True
        return self.activations[idxs]
    
    def refresh(self):
        """
        Refresh the buffer
        """
        print("refreshing buffer...")

        # get rid of read activations
        self.activations = self.activations[~self.read]

        # read in new activations until buffer is full
        idx = 0
        while len(self.activations) < self.n_ctxs * self.ctx_len:
            if idx % self.in_batch_size == 0:
                inputs = []
            text = json.loads(next(self.text_stream))['text']
            inputs.append(text)
            if idx % self.in_batch_size == self.in_batch_size - 1:
                aux = self.model.tokenizer(inputs, return_tensors='pt', max_length=128, padding=True, truncation=True).to(self.device)
                tokens, attention_mask = aux['input_ids'], aux['attention_mask']
                with self.model.generate(max_new_tokens=1, pad_token_id=self.model.tokenizer.pad_token_id) as generator:
                    with generator.invoke(tokens) as invoker:
                        hidden_states = self.submodule.output.save()
                try: # until the bug with the unstable proxy shape is fixed
                    self.activations = t.cat(
                        [self.activations, hidden_states.value[attention_mask == 1].to('cpu')], # activations over non-padding tokens
                        dim=0
                    )
                except:
                    pass
            idx += 1
        
        self.read = t.zeros(len(self.activations)).bool()
        print('buffer refreshed...')

    def get_val_batch(self):
        """
        Get a batch of input texts 
        """
        inputs = [
            json.loads(next(self.text_stream))['text']
            for _ in range(4) # TODO change this back
        ]
        return inputs
        

    def validate(self, ae):
        """
        Get the reconstruction loss on a batch of validation data
        """
        inputs = self.get_val_batch()
        tokens = self.model.tokenizer(inputs, return_tensors='pt', padding=True).input_ids.to(self.device)
        with self.model.generate(max_new_tokens=1) as generator:

            # unmodified logits
            with generator.invoke(inputs) as invoker:
                logits_original = self.model.embed_out.output.save()
            # logits when replacing component output with reconstruction by autoencoder
            with generator.invoke(inputs) as invoker:
                self.submodule.output = ae(self.submodule.output)
                logits_reconstructed = self.model.embed_out.output.save()
            # logits when zero ablating component
            with generator.invoke(inputs) as invoker:
                self.submodule.output = t.zeros_like(self.submodule.output)
                logits_zero = self.model.embed_out.output.save()
            
        # return losses for all three logit types
        outs = []
        for logits in [logits_original, logits_reconstructed, logits_zero]:
            outs.append(t.nn.CrossEntropyLoss()(
                logits.value[:,:-1,:].reshape(-1, logits.value.shape[-1]),
                tokens[:,1:].reshape(-1)
            ).item())
        return tuple(outs)

    def close(self):
        """
        Close the text stream and the underlying compressed file.
        """
        self.text_stream.close()
        
        