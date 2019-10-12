# Credits to the contributors at fast.ai!

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings

def dropout_mask(x, sz, p):
    return x.new(*sz).bernoulli_(1-p).div_(1-p)
  
class RNNDropout(nn.Module):
    """Applies dropout consistent along a time dimension"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p=p

    def forward(self, x):
        if not self.training or self.p == 0.: 
            return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m

class EmbeddingDropout(nn.Module):
    """Dropout for embeddings"""
    def __init__(self, emb, embed_p):
        super(EmbeddingDropout, self).__init__()
        self.emb = emb
        self.embed_p = embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None: self.pad_idx = -1

    def forward(self, words, scale=None):
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0),1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = (self.emb.weight * mask)
        else: 
            masked_embed = (self.emb.weight)
        if scale: 
            masked_embed.mul_(scale)
        out = F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                          self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)
        return out
    
class WeightDropout(nn.Module):
    """Dropconnect for hidden weight matrices in RNNs"""
    def __init__(self, module, weight_p, layer_names=['weight_hh_l0']):
        super(WeightDropout, self).__init__()
        self.module,self.weight_p,self.layer_names = module,weight_p,layer_names
        for layer in self.layer_names:
            #Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        "Apply dropout to the raw weights."
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            #To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()
