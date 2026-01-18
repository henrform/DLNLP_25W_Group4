#!/usr/bin/env bash

TASK_DIR=quantization/glue_data/RTE/
FT_BERT_BASE_DIR=bert_base_uncased_teacher_RTE/
TRAINED_FT_BERT_BASE_DIR=quantization/final_rte_teacher/

python train_teacher.py --data_dir ${TASK_DIR} \
                        --bert_model ${FT_BERT_BASE_DIR} \
                        --output_dir ${TRAINED_FT_BERT_BASE_DIR} \
                        --num_train_epochs 5 \
                        --train_batch_size 32 \
                        --learning_rate 2e-5
