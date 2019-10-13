import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dropout import RNNDropout, EmbeddingDropout, WeightDropout
from utils import repackage_hidden

class RNNModel(nn.Module):
    def __init__(self, encoder, decoder, tie_weights=True, initrange=0.1):
        super(RNNModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        
        if tie_weights:
            self.decoder.fc1.weight = self.encoder.embeddings.weight
            
        # Initialize parameters
        self.encoder.embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder.fc1.bias.data.zero_()
        self.decoder.fc1.weight.data.uniform_(-initrange, initrange)
        
    def reset_hidden(self):
        self.encoder.reset_hidden()
        
    def forward(self, x, **kwargs):
        out = self.decoder(*self.encoder(x), **kwargs)
        return out
    
class RNNClassifier(nn.Module):
    def __init__(self, encoder, decoder):
        super(RNNClassifier, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.groups = ['encoder', 'rnn.0', 'rnn.1', 'rnn.2', 'decoder']
        
    def reset_hidden(self):
        self.encoder.reset_hidden()
        
    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        
    def unfreeze(self, ix):
        to_unfreeze = self.groups[ix:]
        for n, p in self.named_parameters():
            for group in to_unfreeze:
                if group in n: p.requires_grad = True
                    
    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True
        
    def forward(self, x):
        out, hidden, raw_out, dropped_out = self.encoder(x)
        logits = self.decoder(out, hidden[-1])
        return logits

class AWDLSTMEncoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden_dim, num_layers=1, emb_dp=0.1, weight_dp=0.5, input_dp=0.3, hidden_dp=0.3, tie_weights=True, padding_idx=1):
        super(AWDLSTMEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_sz, emb_dim, padding_idx=padding_idx)
        self.emb_dp = EmbeddingDropout(self.embeddings, emb_dp)
        
        self.rnn = nn.ModuleList([nn.LSTM(emb_dim if l == 0 else hidden_dim, (hidden_dim if l != num_layers - 1 else emb_dim) if tie_weights else hidden_dim) for l in range(num_layers)])
        self.weight_dp = nn.ModuleList([WeightDropout(rnn, weight_dp) for rnn in self.rnn])
        self.hidden_dp = RNNDropout(hidden_dp)
        self.input_dp = RNNDropout(input_dp)

        self.hidden, self.cell = None, None
    
    def init_hidden(self, bs):
        weight = next(self.parameters())
        
        hidden = [weight.new_zeros(1, bs, self.rnn[i].hidden_size) for i in range(len(self.rnn))]
        cell  = [weight.new_zeros(1, bs, self.rnn[i].hidden_size) for i in range(len(self.rnn))]
        
        return hidden, cell
    
    def reset_hidden(self):
        self.hidden, self.cell = None, None
    
    def forward(self, x):
        msl, bs = x.shape
        if self.hidden is None and self.cell is None:
            self.hidden, self.cell = self.init_hidden(bs)
        else:
            self.hidden = [repackage_hidden(h) for h in self.hidden]
            self.cell = [repackage_hidden(h) for h in self.cell]

        out = self.emb_dp(x)
        
        raw_output = []
        dropped_output = []
        out = self.input_dp(out)
        for i in range(len(self.rnn)):
            out, (self.hidden[i], self.cell[i]) = self.weight_dp[i](out, (self.hidden[i], self.cell[i]))
            raw_output.append(out)
            
            if i < len(self.rnn) - 1: 
                out = self.hidden_dp(out)
                dropped_output.append(out)

        return out, self.hidden, raw_output, dropped_output

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_sz, emb_dim, hidden_dim, num_layers=1, dropout=0.5):
        super(LSTMEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_sz, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.hidden, self.cell = None, None
    
    def init_hidden(self, bs):
        weight = next(self.parameters())
        nlayers = self.rnn.num_layers
        nhid = self.rnn.hidden_size
        
        return (weight.new_zeros(nlayers, bs, nhid), weight.new_zeros(nlayers, bs, nhid))
    
    def reset_hidden(self):
        self.hidden, self.cell = None, None
    
    def forward(self, x, lens=None):
        msl, bs = x.shape
        
        if self.hidden is None and self.cell is None:
            self.hidden, self.cell = self.init_hidden(bs)
        else:
            self.hidden, self.cell = repackage_hidden((self.hidden, self.cell))

        out = self.embeddings(x)
        out = self.dropout(out)

        out, (self.hidden, self.cell) = self.rnn(out, (self.hidden, self.cell))
        out = self.dropout(out)

        return out, self.hidden, self.cell

class DropoutLinearDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab_sz, out_dp=0.4):
        super(DropoutLinearDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, vocab_sz)
        self.out_dp = RNNDropout(out_dp)
        
    def forward(self, out, hidden, raw, dropped, return_states=False):
        out = self.out_dp(out)
        dropped.append(out)
        out = self.fc1(out)
        
        if return_states:
            return out, hidden, raw, dropped
        return out
    
class LinearDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab_sz):
        super(LinearDecoder, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, vocab_sz)
        
    def forward(self, out, *args, **kwargs):
        return self.fc1(out)

class ConcatPoolingDecoder(nn.Module):
    def __init__(self, hidden_dim, bneck_dim, out_dim, dropout_pool=0.2, dropout_proj=0.1, include_hidden=True):
        super(ConcatPoolingDecoder, self).__init__()
        self.bn1 = nn.BatchNorm1d(hidden_dim * 3 if include_hidden else hidden_dim * 2)
        self.bn2 = nn.BatchNorm1d(bneck_dim)
        self.linear1 = nn.Linear(hidden_dim * 3 if include_hidden else hidden_dim * 2, bneck_dim)
        self.linear2 = nn.Linear(bneck_dim, out_dim)
        self.dropout_pool = nn.Dropout(dropout_pool)
        self.dropout_proj = nn.Dropout(dropout_proj)
        self.include_hidden = include_hidden
    
    def forward(self, out, hidden):
        _, bs, _ = out.shape
        avg_pool = F.adaptive_avg_pool1d(out.permute(1, 2, 0), 1).view(bs, -1)
        max_pool = F.adaptive_max_pool1d(out.permute(1, 2, 0), 1).view(bs, -1)
        if self.include_hidden:
            pooled = torch.cat([hidden[-1], avg_pool, max_pool], dim=1)
        else:
            pooled = torch.cat([avg_pool, max_pool], dim=1)
        out = self.dropout_pool(self.bn1(pooled))
        out = torch.relu(self.linear1(out))
        out = self.dropout_proj(self.bn2(out))
        out = self.linear2(out)
        return out