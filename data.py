import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>': 0, '<pad>': 1, '<eos>': 2}
        self.idx2word = ['<unk>', '<pad>', '<eos>']
        self.vocab_set = set(self.idx2word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.vocab_set.add(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, train_file, valid_file, test_file, load_vocab=False, vocab_file='vocab.pth'):
        self.dictionary = Dictionary()
        if load_vocab:
            with open(os.path.join(path, vocab_file), 'rb') as f:
                word2idx, idx2word = torch.load(f)
            self.dictionary.word2idx = word2idx
            self.dictionary.idx2word = idx2word
            self.dictionary.vocab_set = set(idx2word)
        self.train = self.tokenize(os.path.join(path, train_file), skip_dict=load_vocab)
        self.valid = self.tokenize(os.path.join(path, valid_file), skip_dict=load_vocab)
        self.test = self.tokenize(os.path.join(path, test_file), skip_dict=load_vocab)
        
    def build_dict(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

    def tokenize(self, path, skip_dict=False):
        if not skip_dict:
            self.build_dict(path)
        
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word if word in self.dictionary.vocab_set else '<unk>'])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)

        return ids