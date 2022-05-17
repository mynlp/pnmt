# Pretrain-OpenNMT-py (PNMT): Open-Source Neural Machine Translation with Pre-train support and research friendly feature

This repository is an extension from OpenNMT-py which supports the pre-train model including BERT model or other pre-trained models. The target of this repository is to make OpenNMT a more research friendly project that support pre-train model, auto evaluation and find the best checkpoint on the test set.
Before you use this package, you should refer to [OpenNMT](https://github.com/OpenNMT/OpenNMT-py) for basic usage as this repository is build on OpenNMT. 

However, as this is an independent extension for OpenNMT, so I may not be able to always keep updated with OpenNMT's new release, but I will try my best. If any new feature of OpenNMT does not work in here, you should use OpenNMT instead, but if you think it is a bug in the repository, please raise an issue.
## Completed Features:
### BERT as Embedding
In this feature, BERT works as an embedding layer that provide word embedding given a token. Therefore the BERT is not the encoder at this feature, the RNN or any other model can be chosen to be the encoder.
### BERT as Encoder
In this feature, BERT works an the encoder which makes it a BERT2Seq model.

## Installation

This package is not available in pip as most of the code is still experimental so you should install in from source.
```
git clone https://github.com/PosoSAgapo/Pretrain-OpenNMT-py.git
cd Pretrain-OpenNMT-py
pip install -e .
```
Note: if you encounter a MemoryError during installation, try to use pip with --no-cache-dir. For other installation details, please refer to OpenNMT.

## Example
Before you get to use Pre-train-OpenNMT, as indicated above, you should refer to OpenNMT for basic usage, the use of this repository depends on the OpenNMT, so it would be better if you are familiar with it.
To use bert as embedding model, you only need to specify the embeddings_type argument in the YAML file, for example:
```
save_data: examples/data/example
src_vocab: examples/vocab/example.vocab.src
tgt_vocab: examples/vocab/example.vocab.tgt
overwrite: True
# Corpus opts:
data:
    corpus_1:
        path_src: examples/data/train_src.txt
        path_tgt: examples/data/train_tgt.txt
    valid:
        path_src: examples/data/valid_src.txt
        path_tgt: examples/data/valid_tgt.txt
save_model: examples/run/model
save_checkpoint_steps: 10000
train_steps: 100
valid_steps: 5
embeddings_type: bert-base-uncased
use_pre_trained_model_for_embedding: True
word_vec_size: 768
rnn_size: 384
copy_attn: True
```
After this YAML file is built, instead of running `onmt_train -config xxx.yaml`, you should use `pnmt_train -config xxx.yaml` instead, then you should be able to see the log ouput shows that the generation of bert embedding for both src vocab and tgt vocabulary.

In this example, you specify the `embeddings_type` as the `bert-base-uncased` and the word vector size is `768`, the Pre-train-OpenNMT will autromatically load the tokenizer and model of `bert-base-uncased` based on the transformers package. 

Then, it will generate the word embedding for both of your src and tgt vocabulary, then the pre-train model and the tokenizer will be deleted after the generation to save cuda memeory as the bert only works as embedding not encoder. Basically, Pre-train-OpenNMT works seamlessly with OpenNMT.

Since the whole bert family is supported, you could also specify `bert-large-uncased` or `bert-base-cased` or any other bert version supported by transformers package. 

The dafult embedding for each word in the embedding of `[CLS]` token which is the embedding representation of that word.

Additionally, all the `onmt` script are replaced with `pnmt` is this repository.

## To do features:
### Generation Pre-trained Model 
This include pre-trained models like T5 or other possible generation pre-trained models.
### Parrallel Training and Inference
This feature is mainly to be research friendlym, the target is to split the test and training and then automatically find the best checkpoint.
## Others
This project will be organized and re-publish as another package since OpenNMT does not consider to include pre-trained models.

Feel free to send a PR or feature request, I will reply at my best.

