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

from transformers import WarmupLinearSchedule
from layers import RNNModel, AWDLSTMEncoder, DropoutLinearDecoder, LSTMEncoder, LinearDecoder
from utils import count_parameters, get_loaders
from data import Corpus, Dictionary

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='../data/wikitext-2', help='location of the data corpus')
parser.add_argument('--train', type=str, default='wiki.train.tokens', help='name of the training corpus')
parser.add_argument('--valid', type=str, default='wiki.valid.tokens', help='name of the validation corpus')
parser.add_argument('--test', type=str, default='wiki.test.tokens', help='name of the testing corpus')
parser.add_argument('--output', type=str, default='awd_lstm', help='output name')

parser.add_argument('--bs', type=int, default=80, help='batch size')
parser.add_argument('--eval_bs', type=int, default=10, help='evaluation batch size')
parser.add_argument('--bptt', type=int, default=80, help='bptt length')
parser.add_argument('--use_var_bptt', action='store_true', help='use variable length bptt')
parser.add_argument('--rebuild_dataset', action='store_true', help='force rebuild the dataset')
parser.add_argument('--load_vocab', action='store_true', help='load vocabulary')
parser.add_argument('--vocab_file', type=str, default='vocab.pth', help='pretrained vocabulary file')
parser.add_argument('--save_vocab', action='store_true', help='load vocabulary')

parser.add_argument('--encoder', type=str, default='awd_lstm', choices=['awd_lstm', 'lstm'], help='encoder')
parser.add_argument('--decoder', type=str, default='dropoutlinear', choices=['dropoutlinear', 'linear'], help='decoder')
parser.add_argument('--emb_dim', type=int, default=400, help='embedding dimensions')
parser.add_argument('--hidden_dim', type=int, default=1152, help='hidden dimensions')
parser.add_argument('--num_layers', type=int, default=3, help='number of rnn layers')
parser.add_argument('--emb_dp', type=float, default=0.1, help='embeddng dropout')
parser.add_argument('--hidden_dp', type=float, default=0.3, help='hidden to hidden dropout')
parser.add_argument('--input_dp', type=float, default=0.3, help='input dropout')
parser.add_argument('--weight_dp', type=float, default=0.5, help='dropconnect dropout')
parser.add_argument('--out_dp', type=float, default=0.4, help='output dropout')
parser.add_argument('--initrange', type=float, default=0.05, help='initialization range')
parser.add_argument('--tie_weights', action='store_true', help='tie embeddings and decoder weights')
parser.add_argument('--use_pretrained', action='store_true', help='use pretrained weights')
parser.add_argument('--pretrained_file', type=str, default='pretrained_wt103', help='pretrained model file')

parser.add_argument('--anneal_factor', type=float, default=4.0, help='learning rate anneal rate')
parser.add_argument('--lr', type=float, default=30, help='learning rate')
parser.add_argument('--no_lr_scaling', action='store_true', help='no lr scaling with var bptt or no auto anneal otherwise')
parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimzer to use')
parser.add_argument('--no_warmup', action='store_true', help='do not use linear warmups when using Adam')
parser.add_argument('--warmup_pct', type=float, default=0.1, help='percentage of steps for warmup')

parser.add_argument('--epochs', type=int, default=2, help='epochs to train the network')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--alpha', type=float, default=2.0, help='AR alpha parameter')
parser.add_argument('--beta', type=float, default=1.0, help='TAR beta parameter')

parser.add_argument('--no_cuda', action='store_true', help='do not use CUDA')
parser.add_argument('--gpu', type=int, default=0, help='index of GPU to use')
parser.add_argument('--seed', type=int, default=42, help='random seed')

args = parser.parse_args()
print(args)

if args.decoder == 'dropoutlinear': assert args.encoder == 'awd_lstm'

# CUDA
device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and not args.no_cuda else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed);
torch.cuda.manual_seed(args.seed);
torch.backends.cudnn.deterministic = True

# Produce or load the dataset
path = args.path
fn = '{}/corpus.{}.data'.format(path, hashlib.md5(path.encode()).hexdigest())
if os.path.exists(fn) and not args.rebuild_dataset:
    print('Loading cached dataset...')
    corpus = torch.load(fn)
else:
    print('Producing dataset...')
    if args.load_vocab:
        print('Vocabulary has been loaded from {}'.format(args.vocab_file))
    corpus = Corpus(path, args.train, args.valid, args.test, load_vocab=args.load_vocab, vocab_file=args.vocab_file)
    torch.save(corpus, fn)
    if args.save_vocab:
        with open('{}/{}'.format(path, args.vocab_file), 'wb') as f:
            torch.save([corpus.dictionary.word2idx, corpus.dictionary.idx2word], f)
    
vocab_sz = len(corpus.dictionary)

# Produce dataloaders
train_loader = get_loaders(corpus.train, args.bs, args.bptt, use_var_bptt=args.use_var_bptt)
valid_loader = get_loaders(corpus.valid, args.eval_bs, args.bptt)
test_loader = get_loaders(corpus.test, args.eval_bs, args.bptt)

# Construct encoder
if args.encoder == 'awd_lstm':
    encoder = AWDLSTMEncoder(vocab_sz=vocab_sz, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, 
                             num_layers=args.num_layers, emb_dp=args.emb_dp, weight_dp=args.weight_dp, 
                             input_dp=args.input_dp, hidden_dp=args.hidden_dp, tie_weights=args.tie_weights)
elif args.encoder == 'lstm':
    encoder = LSTMEncoder(vocab_sz=vocab_sz, emb_dim=args.emb_dim, num_layers=args.num_layers, 
                          hidden_dim=args.emb_dim if args.tie_weights else args.hidden_dim, dropout=args.weight_dp)

# Construct decoder    
if args.decoder == 'dropoutlinear':
    decoder = DropoutLinearDecoder(hidden_dim=args.emb_dim if args.tie_weights else args.hidden_dim, 
                                   vocab_sz=vocab_sz, out_dp=args.out_dp)
elif args.decoder == 'linear':
    decoder = LinearDecoder(hidden_dim=args.emb_dim if args.tie_weights else args.hidden_dim, vocab_sz=vocab_sz)
    
# Produce model
model = RNNModel(encoder, decoder, tie_weights=args.tie_weights, initrange=args.initrange).to(device)
print(model)

# Pretrained
if args.use_pretrained:
    print("Using pretrained model {}".format(args.pretrained_file))
    with open('{}/{}'.format(path, args.pretrained_file), 'rb') as f:
        inc = model.load_state_dict(torch.load(f), strict=False)
    print(inc)

# Optimization setup
criterion = nn.CrossEntropyLoss()
optimizer, scheduler = None, None
if args.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
elif args.optimizer == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    steps = len(train_loader) * args.epochs
    if not args.no_warmup:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(steps * args.warmup_pct), t_total=steps)

print("Optimization settings:")
print(optimizer)
print("Scheduler: {}".format(scheduler))
print("The model has {:,} trainable parameters".format(count_parameters(model)))

# Training setup
best_loss = np.inf
best_epoch = 0
train_losses = []
valid_losses = []

# Training!
print("Beginning training")
try:
    for e in range(1, args.epochs + 1):
        model.train()
        model.reset_hidden()
        train_loss = 0
        for batch in tqdm(train_loader):
            x, y = batch

            # Scale learning rate to sequence length
            if args.use_var_bptt and not args.no_lr_scaling:
                seq_len, _ = x.shape
                optimizer.param_groups[0]['lr'] = args.lr * seq_len / args.bptt

            x = x.to(device)
            y = y.to(device)

            out = model(x, return_states=True)
            if args.encoder == 'awd_lstm': out, hidden, raw_out, dropped_out = out
            raw_loss = criterion(out.view(-1, vocab_sz), y)

            # AR/TAR
            loss = raw_loss
            if args.encoder == 'awd_lstm':
                loss += args.alpha * dropped_out[-1].pow(2).mean()
                loss += args.beta * (raw_out[-1][1:] - raw_out[-1][:-1]).pow(2).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            if scheduler is not None: scheduler.step()

            # Restore original LR
            if args.use_var_bptt and not args.no_lr_scaling:
                optimizer.param_groups[0]['lr'] = args.lr

            train_loss += raw_loss.item()
        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        model.reset_hidden()
        valid_loss = 0
        for batch in tqdm(valid_loader):
            with torch.no_grad():
                x, y = batch
                x = x.to(device)
                y = y.to(device)

                out = model(x)
                loss = criterion(out.view(-1, vocab_sz), y)

                valid_loss += loss.item()
        valid_loss /= len(valid_loader)
        valid_losses.append(valid_loss)

        # Track and anneal LR
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_epoch = e
            print("Best loss so far. Saving model.")
            with open('{}/{}.pth'.format(path, args.output), 'wb') as f:
                torch.save(model.state_dict(), f)
        else:
            if not args.use_var_bptt and not args.no_lr_scaling:
                optimizer.param_groups[0]['lr'] /= args.anneal_factor
        cur_lr = optimizer.param_groups[0]['lr']

        print("Epoch {:3} | Train Loss {:.4f} | Train Ppl {:.4f} | Valid Loss {:.4f} | Valid Ppl {:.4f} | LR {:.4f}".format(e, train_loss, np.exp(train_loss), valid_loss, np.exp(valid_loss), cur_lr))

except KeyboardInterrupt:
    print("Exiting training early")

# Load best saved model
print("Loading best model")
with open('{}/{}.pth'.format(path, args.output), 'rb') as f:
    model.load_state_dict(torch.load(f))

# Testing evaluation
print("Evaluating model")
model.eval()
model.reset_hidden()
test_loss = 0
for batch in tqdm(test_loader):
    with torch.no_grad():
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        out = model(x)
        loss = criterion(out.view(-1, vocab_sz), y)

        test_loss += loss.item()
test_loss /= len(test_loader)

print("Test Loss {:.4f} | Test Ppl {:.4f}".format(test_loss, np.exp(test_loss)))

# Saving graphs
print("Saving loss data")
pd.DataFrame(data={'train': train_losses, 'valid': valid_losses}).to_csv('{}/{}.csv'.format(path, args.output), index=False)
with open('{}/{}.txt'.format(path, args.output), 'w') as f:
    f.write("Best loss {:.4f} | Best ppl {:.4f} | Epoch {} | Test loss {:.4f} | Test ppl {:.4f}".format(best_loss, np.exp(best_loss), best_epoch, test_loss, np.exp(test_loss)))
