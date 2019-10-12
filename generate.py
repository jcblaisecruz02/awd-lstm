import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from io import open
import hashlib
import argparse

from layers import RNNModel, AWDLSTMEncoder, DropoutLinearDecoder
from utils import count_parameters, get_loaders
from data import Corpus, Dictionary

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/wikitext-2', help='location of the data corpus')
parser.add_argument('--train', type=str, default='wiki.train.tokens', help='name of the training corpus')
parser.add_argument('--valid', type=str, default='wiki.valid.tokens', help='name of the validation corpus')
parser.add_argument('--test', type=str, default='wiki.test.tokens', help='name of the testing corpus')
parser.add_argument('--output', type=str, default='awd_lstm_checkpoint', help='output name')

parser.add_argument('--emb_dim', type=int, default=400, help='embedding dimensions')
parser.add_argument('--hidden_dim', type=int, default=1150, help='hidden dimensions')
parser.add_argument('--num_layers', type=int, default=3, help='number of rnn layers')
parser.add_argument('--emb_dp', type=float, default=0.1, help='embeddng dropout')
parser.add_argument('--hidden_dp', type=float, default=0.3, help='hidden to hidden dropout')
parser.add_argument('--input_dp', type=float, default=0.3, help='input dropout')
parser.add_argument('--weight_dp', type=float, default=0.5, help='dropconnect dropout')
parser.add_argument('--out_dp', type=float, default=0.4, help='output dropout')
parser.add_argument('--initrange', type=float, default=0.05, help='initialization range')
parser.add_argument('--tie_weights', action='store_true', help='tie embeddings and decoder weights')

parser.add_argument('--use_cuda', action='store_true', help='use CUDA')
parser.add_argument('--gpu', type=int, default=0, help='index of GPU to use')
parser.add_argument('--seed', type=int, default=42, help='random seed')

parser.add_argument('--word', type=str, default='why', help='starting word')
parser.add_argument('--temp', type=float, default=1.0, help='temperature')
parser.add_argument('--nwords', type=int, default=100, help='number of words to generate')

args = parser.parse_args()

print(args)

# CUDA
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.use_cuda else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed);
torch.cuda.manual_seed(args.seed);
torch.backends.cudnn.deterministic = True

# Produce or load the dataset
path = args.path
fn = '{}/corpus.{}.data'.format(path, hashlib.md5(path.encode()).hexdigest())
if os.path.exists(fn):
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    corpus = Corpus(path, args.train, args.valid, args.test)
    torch.save(corpus, fn)
    
vocab_sz = len(corpus.dictionary)

# Recreate the model
print("Producing model")
encoder = AWDLSTMEncoder(vocab_sz=vocab_sz, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, 
                         num_layers=args.num_layers, emb_dp=args.emb_dp, weight_dp=args.weight_dp, 
                         input_dp=args.input_dp, hidden_dp=args.hidden_dp, tie_weights=args.tie_weights)
decoder = DropoutLinearDecoder(hidden_dim=args.emb_dim if args.tie_weights else args.hidden_dim, 
                               vocab_sz=vocab_sz, out_dp=args.out_dp)
model = RNNModel(encoder, decoder, tie_weights=args.tie_weights, initrange=args.initrange).to(device)
print(model)

# Load the weights
print("Loading pretrained model")
with open('{}/{}.pth'.format(path, args.output), 'rb') as f:
    model.load_state_dict(torch.load(f))

# Pick starting word
word = args.word
ix = corpus.dictionary.word2idx[word if word in corpus.dictionary.word2idx else '<unk>'] 
inp = torch.LongTensor([ix]).unsqueeze(0).to(device)

# Generate
print(word, end=' ')
model.reset_hidden()
with torch.no_grad():  # no tracking history
    for i in range(args.nwords):
        output = model(inp)
        word_weights = output.squeeze().div(args.temp).exp().cpu()
        word_idx = torch.multinomial(word_weights, 1)[0]
        inp.fill_(word_idx)

        word = corpus.dictionary.idx2word[word_idx]
        print(word, end=' ')
print()