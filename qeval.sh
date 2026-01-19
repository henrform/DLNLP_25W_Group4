#!/usr/bin/env bash

# Runs evaluation of (dynamically) quantized models as an alternative to GPTQ quantization.
# This is because both 4 and 8 bit quantized models via GPTQ resulted in the *exact* same
# accuracy for RTE (~47%).

TINYBERT_DIR=quantization/dynamic_8_bits_distilled_tinybert_6l_rte/
TASK_DIR=quantization/glue_data/RTE/
TASK_NAME=RTE
OUTPUT_DIR=quantization/results/dynamic_8_bits_tinybert_6l_rte/

# ${TINYBERT_DIR} includes the config file, student model and vocab file.

python qtask_distill.py --do_eval \
                       --student_model ${TINYBERT_DIR} \
                       --data_dir ${TASK_DIR} \
                       --task_name ${TASK_NAME} \
                       --output_dir ${OUTPUT_DIR} \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128 \
                       --no_cuda
                                   