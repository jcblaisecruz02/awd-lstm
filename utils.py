import torch
import numpy as np

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def get_loaders(source, bs, bptt, use_var_bptt=False):
    data = batchify(source, bs)
    loader = []
    
    i = 0
    while i < data.size(0) - 2:
        if use_var_bptt:
            rand_bptt = bptt if np.random.random() < 0.95 else bptt / 2.
            seq_len = max(5, int(np.random.normal(rand_bptt, 5)))
        else:
            seq_len = bptt
        
        batch = get_batch(data, i, seq_len)
        loader.append(batch)
        
        i += seq_len
    
    return loader

def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)