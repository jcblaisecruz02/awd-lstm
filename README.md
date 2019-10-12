# AWD-LSTM
Partial reproduction of the paper [Regularizing and Optimizing LSTM Language Models](https://arxiv.org/abs/1708.02182) (Merity et al., 2017). Requires PyTorch 1.0 or greater. 

# Usage
To train a language model on WikiText-2 (or any dataset), the following command may be used:
```
python awd_lstm/main.py \
--path=data/wikitext-2 \
--train=wiki.train.tokens \
--valid=wiki.valid.tokens \
--test=wiki.test.tokens \
--output=pretrained_wt2 \
--bs=80 \
--bptt=80 \
--epochs=500 \
--use_var_bptt \
--tie_weights \
--use_cuda \
--gpu=0
```

# Changelog and To-do
Things on the to-do list:
* Add NT-ASGD support
* Benchmark with new models
* Add more RNN variants (Recurrent Highway networks, etc)