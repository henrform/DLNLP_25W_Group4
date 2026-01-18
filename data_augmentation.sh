#!/usr/bin/env bash

python data_augmentation.py --pretrained_bert_model quantization/bert_base_uncased \
                            --glove_embs quantization/glove_embeddings/glove.6B.300d.txt \
                            --glue_dir quantization/glue_data \
                            --task_name RTE