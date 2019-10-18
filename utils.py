import torch
import torch.utils.data as data_utils
import numpy as np
from tqdm import tqdm

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
    
def drop_mult(model, dm):
    for i in range(len(model.encoder.rnn)):
        model.encoder.weight_dp[i].weight_p *= dm
    model.encoder.emb_dp.embed_p *= dm
    model.encoder.hidden_dp.p *= dm
    model.encoder.input_dp.p *= dm
    return model
    
def accuracy(out, y):
    return torch.sum(torch.max(torch.softmax(out, dim=1), dim=1)[1] == y).item() / len(y)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def vectorize(text, word2idx, vocab_set, msl):
    v_text = [word2idx[word] if word in vocab_set else word2idx['<unk>'] for word in text]
  
    v_text = v_text[:msl]

    if len(v_text) < msl:
        v_text = v_text + [word2idx['<pad>'] for _ in range(msl - len(v_text))]
  
    return v_text

def produce_dataloaders(X_train, y_train, X_val, y_val, word2idx, vocab_set, msl, bs, drop_last=True):
    X_train =  [vectorize(text, word2idx, vocab_set, msl) for text in tqdm(X_train)]
    X_val =  [vectorize(text, word2idx, vocab_set, msl) for text in tqdm(X_val)]
    
    X_train = torch.LongTensor(X_train)
    X_val = torch.LongTensor(X_val)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    
    train_set = data_utils.TensorDataset(X_train, y_train)
    val_set = data_utils.TensorDataset(X_val, y_val)

    train_loader = data_utils.DataLoader(train_set, bs, drop_last=drop_last)
    val_loader = data_utils.DataLoader(val_set, bs, drop_last=drop_last)
    
    return train_loader, val_loader