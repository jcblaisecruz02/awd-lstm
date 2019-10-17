# AWD-LSTM and ULMFiT Toolkit
**Note: This is a work in progress!**

This repository contains partial (so far) reproductions of the following papers:
* [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182) (Merity et al., 2017).
* [Universal Language Model Finetuning for Text Classification](https://arxiv.org/abs/1801.06146) (Howard & Ruder, 2018) 

Code in this repository allows you to:
1. Train AWD-LSTM language models;
2. Finetune language models on other datasets; and
3. Finetune language models for text classification (ULMFiT)

In addition, you can use the layers written in ```layers.py``` for your own work. Details are provided below.

This repository is a work in progress and so not all techniques have been added. Please see the **To-do** section below to see what has not been added yet. Below is also a tracker on the current reproduced scores compared to the papers' original scores.

**Current Results**
* AWD-LSTM on WikiText-2 - Current Test PPL: 74.2074 / **Target Test PPL: 65.8** / Difference: 8.4074
* ULMFiT on iMDB - Current Accuracy: 92.11% / **Target Accuracy: 95.4%** / Difference: 3.29

# Requirements
Libraries you need:
* PyTorch - At least v.1.0.0
* [Transformers](https://github.com/huggingface/transformers) - The learning rate scheduler we use comes from there
* spacy - We use spacy tokenizers for ULMFiT
* numpy and pandas - For random numbers and data manipulation
* tqdm - Progress bars to keep our sanity intact :)

Hardware you need:
* A GPU with at least 11 GB - All the results in this repository have been produced on machines with NVIDIA GTX 1080Ti and servers with NVIDIA Tesla P100 GPUs. A lot of the models when training do take about 10 GB of memory. You *could* get away with a smaller and slower GPU but do note that this affects your speed and performance.

# Using the Training Scripts for Language Modeling
The scripts can be used for language modeling, following the ideas proposed in Merity et al. (2017). Here are a few examples on WikiText-2:

Train an AWD-LSTM using SGD with variable BPTT lengths (Valid PPL: 77.8227 / Test PPL: 74.2074)
```
python awd-lstm/main.py --path=data/wikitext-2 --train=wiki.train.tokens --valid=wiki.valid.tokens --test=wiki.test.tokens --output=pretrained_wt2 --bs=80 --bptt=80 --epochs=150 --use_var_bptt --tie_weights --save_vocab --vocab_file=vocab.pth --gpu=0
```

Train an AWD-LSTM using Adam+LinearWarmup with variable BPTT lengths (Valid PPL: 85.3083 / Test PPL: 80.9872)
```
python awd-lstm/main.py --path=data/wikitext-2 --train=wiki.train.tokens --valid=wiki.valid.tokens --test=wiki.test.tokens --output=pretrained_wt2 --bs=80 --bptt=80 --epochs=150 --use_var_bptt --tie_weights --optimizer=adam --lr=3e-4 --warmup_pct=0.1 --save_vocab --vocab_file=vocab.pth --gpu=0
```

Alternatively, you can test language modeling by training a simple LSTM baseline. Here is an example:

Train a basic LSTM using SGD without variable BPTT lengths (Valid PPL: 101.1345 / Test PPL: 95.7615)
```
python awd-lstm/main.py --path=data/wikitext-2 --train=wiki.train.tokens --valid=wiki.valid.tokens --test=wiki.test.tokens --output=basic_wt2 --bs=80 --bptt=80 --epochs=100 --tie_weights --encoder=lstm --decoder=linear --lr=20 --save_vocab --vocab_file=vocab.pth --gpu=0

```

This is also how language model pretraining works. Be sure to use the ```--save_vocab``` argument to save your vocabularies.

# Generation
You can use your pretrained models to generate text. Here is an example:
```
python awd-lstm/generate.py --path data/wikitext-2 --tie_weights --vocab_file=vocab.pth --pretrained_file=pretrained_wt103.pth --temp=0.7 --nwords=100
```

# Finetuning Language Models
To finetune on a dataset, you'll need the saved vocabulary file and the pretrained weights. For text datasets, you will need to preprocess them such that each sample is separated by a blank line (the code replaces this with an ```<eos>``` token) and each sample has been pre-tokenized (the code should only need to do ```.split()``` to produce tokens). Here's an example finetuning the iMDB dataset on a pretrained model trained using WikiText-103:

```
python awd-lstm/main.py --path=data/imdb --train=train.txt --valid=valid.txt --test=test.txt --output=imdb_finetuned --bs=60 --bptt=60 --epochs=10 --use_var_bptt --tie_weights --load_vocab --vocab_file=vocab.pth --use_pretrained --pretrained_file=pretrained_wt103.pth --gpu=0
```

If you need an example for how to preprocess data, I provide a version of the iMDB Sentiments dataset [here](https://www.kaggle.com/jcblaise/imdb-sentiments). The .csv files are for classification and the .txt files are for language model finetuning.

I cannot, at the moment, provide my own pretrained WikiText-103 models. For the results involving pretrained models, I adapted the pretrained weights provided by [FastAI](https://www.fast.ai/) for now (compatible checkpoint [here](https://storage.googleapis.com/blaisecruz/ulmfit/pretrained_wt103.zip)). More details are in the **To-do** section below.

# ULMFiT / Finetuning Classifiers
To finetune a classifier, make sure you have a finetuned language model at hand. Load the ```ULMFiT.ipynb``` notebook and follow the instructions there.

# Using Layers
The ```layers.py``` file provides some layers you can use outside of this project.

Included so far are:
* Two encoder layers - ```AWDLSTMEncoder``` and ```LSTMEncoder```
* Two language modeling decoder layers - ```DropoutLinearDecoder``` and ```LinearDecoder```
* One classification decoder layer - ```ConcatPoolingDecoder```

For language modeling, a ```RNNModel``` wrapper is provided. Pass in an encoder and a decoder and you're good to go. For example:

```python
encoder = AWDLSTMEncoder(vocab_sz=vocab_sz, emb_dim=400, hidden_dim=1152, num_layers=3, tie_weights=True)
decoder = DropoutLinearDecoder(hidden_dim=400, vocab_sz=vocab_sz) # Using 400 since we're tying weights
model = RNNModel(encoder, decoder, tie_weights=True).to(device)

# Your code here
```

The encoders in the API are written as standalone and independent - they can ouput any number of parameters. The decoders in the API are written to be able to handle any number of parameters given to them. This allows the ```RNNModel``` to act like a wrapper - you can mix and match encoders and decoders. Please refer to ```layers.py``` for more information

Classification is very similar, using the ```RNNClassifier``` wrapper. For example:

```python
encoder = AWDLSTMEncoder(vocab_sz=vocab_sz, emb_dim=400, hidden_dim=1152, num_layers=3)
decoder = ConcatPoolingDecoder(hidden_dim=400, bneck_dim=50, out_dim=2)
model = RNNClassifier(encoder, decoder).to(device)

# Your code here
```

The ```RNNClassifier``` wrapper also has class functions for freezing and unfreezing layers (for ULMFiT). For now, it's only tested to work without errors with the ```AWDLSTMEncoder```, since it was designed for it.

The encoders and decoders can also be used without the ```RNNModel``` and ```RNNClassifier``` wrappers, should you want to. You can use them inside your own models, like the ```AWDLSTMEncoder``` for example:

```python

class MyModel(nn.Module):
    def __init__(self, ...):
        super(MyModel, self).__init__()
        self.encoder = AWDLSTMEncoder(...)

        # Your code here

```

# Changelog
Version 0.3
* Added ULMFiT and related code
* Added finetuning and pretraining capabilities
* Added a way to load vocabularies
* Fixed the generation script

Version 0.2
* Added basic LSTM Encoder support and modularity
* Added an option to train with Adam with Linear Warmups
* Fixed parameters and reduced required parameters
* AR/TAR only activates when using an AWD-LSTM

Version 0.1
* Added basic training functionality

# To-do and Current Progress
As said, this repository is a work in progress. There will be some features missing. Here are the missing features:

For AWD-LSTM training:
* NT-ASGD not implemented

For ULMFiT:
* Discriminative learning rates
* STLR*

Miscellany
* Distributed training

For now, ULMFiT achieves a validation accuracy of 92.11%, which is 3.29 points below the paper's original result of 95.4%. I surmise that this score will get closer once I add in all the missing features. For now, the repo is a partial reproduction.

The pretrained WikiText-103 model used in the results was also adapted from FastAI. I will update with newer scores once I add in distributed pretraining to train my own language model.

\* *The finetuning code prefers Linear Warmup Schedulers over Slanted Triangular Learning Rate Schedulers, but for completion's sake, I will add STLR as an option.*

# Credits
Credits are due to the following:
* The contributors at FastAI where I adapted the dropout code from.
* The people at HuggingFace responsible for their amazing Transformers library.

# Issues and Contributing
If you find an issue, please do let me know in the issues tab! Help and suggestions are also very much welcome.
