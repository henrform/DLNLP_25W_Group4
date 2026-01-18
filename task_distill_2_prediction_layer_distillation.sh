#!/usr/bin/env bash

FT_BERT_BASE_DIR=bert_base_uncased_teacher_RTE/
TMP_TINYBERT_DIR=quantization/tmp_tinybert_task_distill/
TASK_DIR=quantization/glue_data/RTE/
TASK_NAME=RTE
TINYBERT_DIR=quantization/tinybert_task_distill

python task_distill.py --pred_distill  \
                       --teacher_model ${FT_BERT_BASE_DIR} \
                       --student_model ${TMP_TINYBERT_DIR} \
                       --data_dir ${TASK_DIR} \
                       --task_name ${TASK_NAME} \
                       --output_dir ${TINYBERT_DIR} \
                       --aug_train  \
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --eval_step 100 \
                       --max_seq_length 128 \
                       --train_batch_size 32
