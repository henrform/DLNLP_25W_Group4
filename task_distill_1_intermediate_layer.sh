#!/usr/bin/env bash

# ${FT_BERT_BASE_DIR}$ contains the fine-tuned BERT-base model.

FT_BERT_BASE_DIR=bert_base_uncased_teacher_RTE/
GENERAL_TINYBERT_DIR=quantization/tinybert_general_4l/
TASK_DIR=quantization/glue_data/RTE/
TASK_NAME=RTE
TMP_TINYBERT_DIR=quantization/tmp_tinybert_task_distill

python task_distill.py --teacher_model ${FT_BERT_BASE_DIR} \
                       --student_model ${GENERAL_TINYBERT_DIR} \
                       --data_dir ${TASK_DIR} \
                       --task_name ${TASK_NAME} \
                       --output_dir ${TMP_TINYBERT_DIR} \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --num_train_epochs 10 \
                       --aug_train \
                       --do_lower_case  
                         