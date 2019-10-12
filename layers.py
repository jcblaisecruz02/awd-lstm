import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dropout import RNNDropout, EmbeddingDropout, WeightDropout
from utils import repackage_hidden

class RNNModel(nn.Module):
    """
    Wrapper for language models. Accepts an encoder and a decoder module.
    Keyword arguments are passed on to the decoder, which also accepts
    all the encoder outputs.
    """
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

class AWDLSTMEncoder(nn.Module):
    """
    AWD-LSTM Encoder module
    """
    def __init__(self, vocab_sz, emb_dim, hidden_dim, num_layers=1, emb_dp=0.1, weight_dp=0.5, input_dp=0.3, hidden_dp=0.3, tie_weights=True):
        super(AWDLSTMEncoder, self).__init__()
        self.embeddings = nn.Embedding(vocab_sz, emb_dim)
        self.emb_dp = EmbeddingDropout(self.embeddings, emb_dp)
        
        self.rnn = nn.ModuleList([nn.LSTM(emb_dim if l == 0 else hidden_dim, (hidden_dim if l != num_layers - 1 else emb_dim) if tie_weights else hidden_dim) for l in range(num_layers)])
        self.weight_dp = nn.ModuleList([WeightDropout(rnn, weight_dp) for rnn in self.rnn])
        self.hidden_dp = RNNDropout(hidden_dp)
        self.input_dp = RNNDropout(input_dp)

        self.hidden, self.cell = None, None
    
    def init_hidden(self, bs):
        """Initializes a list of hidden and cell states."""
        weight = next(self.parameters())
        
        hidden = [weight.new_zeros(1, bs, self.rnn[i].hidden_size) for i in range(len(self.rnn))]
        cell  = [weight.new_zeros(1, bs, self.rnn[i].hidden_size) for i in range(len(self.rnn))]
        
        return hidden, cell
    
    def reset_hidden(self):
        """Resets hidden states to None. Use at the start of every epoch."""
        self.hidden, self.cell = None, None
    
    def forward(self, x, lens=None):
        msl, bs = x.shape

        # Get initial hidden/cell or detatch the current one from history
        if self.hidden is None and self.cell is None:
            self.hidden, self.cell = self.init_hidden(bs)
        else:
            self.hidden = [repackage_hidden(h) for h in self.hidden]
            self.cell = [repackage_hidden(h) for h in self.cell]

        # Embed the inputs
        out = self.emb_dp(x)
        
        # Track the raw and dropped-out outputs per layaer of RNN
        # then apply RNN dropout to the embedded inputs
        raw_output = []
        dropped_output = []
        out = self.input_dp(out)

        # Take new hiddens and cells per layer of RNN, applying dropout in between layers
        for i in range(len(self.rnn)):
            out, (self.hidden[i], self.cell[i]) = self.weight_dp[i](out, (self.hidden[i], self.cell[i]))
            raw_output.append(out)
            
            if i < len(self.rnn) - 1: 
                out = self.hidden_dp(out)
                dropped_output.append(out)

        return out, self.hidden, raw_output, dropped_output
    
class DropoutLinearDecoder(nn.Module):
    """
    Linear Decoder with RNN dropout
    """
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
