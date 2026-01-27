#!/usr/bin/env bash

# TINYBERT_DIR=quantization/dynamic_8_bits_distilled_tinybert_6l_rte/
# OUTPUT_DIR=quantization/results/dynamic_8_bits_tinybert_6l_rte/

# TINYBERT_DIR=quantization/quantized_8_bits_distilled_tinybert_6l_rte/
# OUTPUT_DIR=quantization/results/quantized_8_bits_tinybert_6l_rte/

TINYBERT_DIR=quantization/distilled_tinybert_6l_rte/
OUTPUT_DIR=quantization/results/distilled_tinybert_6l_rte/

TASK_DIR=quantization/glue_data/RTE/
TASK_NAME=RTE

# ${TINYBERT_DIR} includes the config file, student model and vocab file.

python task_distill.py --do_eval \
                       --student_model ${TINYBERT_DIR} \
                       --data_dir ${TASK_DIR} \
                       --task_name ${TASK_NAME} \
                       --output_dir ${OUTPUT_DIR} \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128
                                   