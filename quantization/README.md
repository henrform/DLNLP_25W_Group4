# Quantization

## Setup

Download pre-trained Bert model as well as general TinyBERT models:

```shell
python download_model.py
```

Download and unzip GloVe embeddings used for data augmentation:

```shell
# Wikipedia 2014 + Gigaword 5 (6B tokens, 400K vocab, uncased, 50d, 100d, 200d, & 300d vectors, 822 MB download):
curl -sSLO https://nlp.stanford.edu/data/glove.6B.zip

mkdir -p glove_embeddings

unzip glove.6B.zip -d glove_embeddings
```

Download all GLUE data:

```shell
python download_glue_data.py
```

## Data Augmentation

Run data augmentation for GLUE task (e.g. SST-2):

```shell
python data_augmentation.py --pretrained_bert_model quantization/bert_base_uncased \
                            --glove_embs quantization/glove_embeddings/glove.6B.300d.txt \
                            --glue_dir quantization/glue_data \
                            --task_name SST-2
```