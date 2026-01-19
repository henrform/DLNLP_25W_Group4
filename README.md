Reproduction
============

This is a clone of the code provided by the original Authors of TinyBERT. You can find the original code [here](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) and the paper [here](https://arxiv.org/abs/1909.10351). We tried to preserve the original codebase as much as possible. Some additional scripts are necessary to make the original code work. We still had to modify some parts where the script crashed, most likely due to an old Python version.

## Setup and get required data

First, we need to create a conda environment and activate it
```bash
conda env create -f environment.yml
conda activate DLNLP_25W_Group4
```

Next, setup folder structure
```bash
mkdir data models
```

Get GloVe data
```bash
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
mv glove.840B.300d.zip data/
unzip data/glove.840B.300d.zip
```

This will download the GLUE data and move it to the data folder
```bash
python download_glue.py
mv glue_data data/
```

Download the English Wikipedia dump from [here](https://meta.wikimedia.org/wiki/Data_dump_torrents#English_Wikipedia) and put it in the data folder. We used the 20251101 version.
```bash
python -m wikiextractor.WikiExtractor data/enwiki-20251101-pages-articles-multistream.xml.bz2 --json -o data/extracted_wiki
```

We need to reformat the data for the script provided by the authors of the original paper. They require one sentence per line and documents to separated by an empty line
```bash
python prepare_data.py
```

Here we download all BERT models required for this reproduction. Note that we only reproduced the results of RTE, SST-2, CoLA and MRPC due to time constraints.
```bash
python get_bert_models.py
```

## Actual reproduction
All the following scripts are from the original repository, except for `fix_teacher_layers.py` and `finetune_teacher.py`.

### General distillation
This will reformat the wikipedia corpus from before for general distillation
```bash
python pregenerate_training_data.py --train_corpus wiki_for_bert.txt --bert_model models/bert-base-uncased --reduce_memory --do_lower_case --epochs_to_generate 3 --output_dir data/json_wiki 
```

The actual general distillation was run with these commands
```bash
python general_distill.py --pregenerated_data json_wiki --teacher_model models/bert-base-uncased --student_model tinybert_4l_config --reduce_memory --do_lower_case --train_batch_size 256 --gradient_accumulation_steps 8 --output_dir models/distilled_tinybert_4l --eval_step 10000
python general_distill.py --pregenerated_data json_wiki --teacher_model models/bert-base-uncased --student_model tinybert_6l_config --reduce_memory --do_lower_case --train_batch_size 256 --gradient_accumulation_steps 8 --output_dir models/distilled_tinybert_6l --eval_step 10000
```

### Task-specific distillation

The following commands will augment the training data as described in the paper
```bash
python data_augmentation.py --pretrained_bert_model models/bert-base-uncased --glove_embs data/glove.840B.300d.txt --glue_dir data/glue_data --task_name RTE
python data_augmentation.py --pretrained_bert_model models/bert-base-uncased --glove_embs data/glove.840B.300d.txt --glue_dir data/glue_data --task_name SST-2
python data_augmentation.py --pretrained_bert_model models/bert-base-uncased --glove_embs data/glove.840B.300d.txt --glue_dir data/glue_data --task_name CoLA
python data_augmentation.py --pretrained_bert_model models/bert-base-uncased --glove_embs data/glove.840B.300d.txt --glue_dir data/glue_data --task_name MRPC
```

For the original code to work, we need `pytorch_model.bin` and not `model.safetensors`. However, when specifying that, we lose the prediction head and the layers get renamed. The first script fixes the names, while the other fine-tunes the teacher again. Since the teachers are already fine-tuned and the prediction head is a binary classification, a short fine-tuning suffices.
```bash
python fix_teacher_layers.py
python finetune_teacher.py
```

The 8 commands below will run the task-specific distillation the 4 and 6 layer TinyBERT
```bash
# 4 layer task-specific distillation
python task_distill.py --teacher_model models/final_rte_teacher --student_model models/distilled_tinybert_4l --data_dir data/glue_data/RTE --task_name RTE --output_dir models/tmp_distilled_tinybert_4l --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill --teacher_model models/final_rte_teacher --student_model models/tmp_distilled_tinybert_4l --data_dir data/glue_data/RTE --task_name RTE --output_dir models/distilled_tinybert_4l_rte --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_cola_teacher --student_model models/distilled_tinybert_4l --data_dir data/glue_data/CoLA --task_name cola --output_dir models/tmp_distilled_tinybert_4l_cola --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_cola_teacher --student_model models/tmp_distilled_tinybert_4l_cola --data_dir data/glue_data/CoLA --task_name cola --output_dir models/distilled_tinybert_4l_cola --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_sst2_teacher --student_model models/distilled_tinybert_4l --data_dir data/glue_data/SST-2 --task_name sst-2 --output_dir models/tmp_distilled_tinybert_4l_sst2 --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_sst2_teacher --student_model models/tmp_distilled_tinybert_4l_sst2 --data_dir data/glue_data/SST-2 --task_name sst-2 --output_dir models/distilled_tinybert_4l_sst2 --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_mrpc_teacher --student_model models/distilled_tinybert_4l --data_dir data/glue_data/MRPC --task_name mrpc --output_dir models/tmp_distilled_tinybert_4l_mrpc --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_mrpc_teacher --student_model models/tmp_distilled_tinybert_4l_mrpc --data_dir data/glue_data/MRPC --task_name mrpc --output_dir models/distilled_tinybert_4l_mrpc --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32


# 6 layer task-specific distillation
python task_distill.py --teacher_model models/final_rte_teacher --student_model models/distilled_tinybert_6l --data_dir data/glue_data/RTE --task_name RTE --output_dir models/tmp_distilled_tinybert_6l --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill --teacher_model models/final_rte_teacher --student_model models/tmp_distilled_tinybert_6l --data_dir data/glue_data/RTE --task_name RTE --output_dir models/distilled_tinybert_6l_rte --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_cola_teacher --student_model models/distilled_tinybert_6l --data_dir data/glue_data/CoLA --task_name cola --output_dir models/tmp_distilled_tinybert_6l_cola --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_cola_teacher --student_model models/tmp_distilled_tinybert_6l_cola --data_dir data/glue_data/CoLA --task_name cola --output_dir models/distilled_tinybert_6l_cola --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_sst2_teacher --student_model models/distilled_tinybert_6l --data_dir data/glue_data/SST-2 --task_name sst-2 --output_dir models/tmp_distilled_tinybert_6l_sst2 --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_sst2_teacher --student_model models/tmp_distilled_tinybert_6l_sst2 --data_dir data/glue_data/SST-2 --task_name sst-2 --output_dir models/distilled_tinybert_6l_sst2 --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_mrpc_teacher --student_model models/distilled_tinybert_6l --data_dir data/glue_data/MRPC --task_name mrpc --output_dir models/tmp_distilled_tinybert_6l_mrpc --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_mrpc_teacher --student_model models/tmp_distilled_tinybert_6l_mrpc --data_dir data/glue_data/MRPC --task_name mrpc --output_dir models/distilled_tinybert_6l_mrpc --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32
```

### Evaluation

Running the commands below will yield the evaluation results. We achieved slightly higher scores, probably due to the more recent wikipedia data.
```bash
# 4 layer evaluation
python task_distill.py --do_eval --student_model models/distilled_tinybert_4l_rte --data_dir data/glue_data/RTE --task_name rte --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/distilled_tinybert_4l_cola --data_dir data/glue_data/CoLA --task_name cola --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/distilled_tinybert_4l_sst2 --data_dir data/glue_data/SST-2 --task_name sst2 --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/distilled_tinybert_4l_mrpc --data_dir data/glue_data/MRPC --task_name mrpc --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp

# 6 layer evaluation
python task_distill.py --do_eval --student_model models/distilled_tinybert_6l_rte --data_dir data/glue_data/RTE --task_name rte --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/distilled_tinybert_6l_cola --data_dir data/glue_data/CoLA --task_name cola --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/distilled_tinybert_6l_sst2 --data_dir data/glue_data/SST-2 --task_name sst2 --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/distilled_tinybert_6l_mrpc --data_dir data/glue_data/MRPC --task_name mrpc --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
```

TinyBERT
======== 
TinyBERT is 7.5x smaller and 9.4x faster on inference than BERT-base and achieves competitive performances in the tasks of natural language understanding. It performs a novel transformer distillation at both the pre-training and task-specific learning stages. The overview of TinyBERT learning is illustrated as follows: 
<br />
<br />
<img src="tinybert_overview.png" width="800" height="210"/>
<br />
<br />

For more details about the techniques of TinyBERT, refer to our paper:

[TinyBERT: Distilling BERT for Natural Language Understanding](https://arxiv.org/abs/1909.10351)


Release Notes
=============
First version: 2019/11/26
Add Chinese General_TinyBERT: 2021.7.27

Installation
============
Run command below to install the environment(**using python3**)
```bash
pip install -r requirements.txt
```

General Distillation
====================
In general distillation, we use the original BERT-base without fine-tuning as the teacher and a large-scale text corpus as the learning data. By performing the Transformer distillation on the text from general domain, we obtain a general TinyBERT which provides a good initialization for the task-specific distillation. 

General distillation has two steps: (1) generate the corpus of json format; (2) run the transformer distillation;

Step 1: use `pregenerate_training_data.py` to produce the corpus of json format  


```
 
# ${BERT_BASE_DIR}$ includes the BERT-base teacher model.
 
python pregenerate_training_data.py --train_corpus ${CORPUS_RAW} \ 
                  --bert_model ${BERT_BASE_DIR}$ \
                  --reduce_memory --do_lower_case \
                  --epochs_to_generate 3 \
                  --output_dir ${CORPUS_JSON_DIR}$ 
                             
```

Step 2: use `general_distill.py` to run the general distillation
```
 # ${STUDENT_CONFIG_DIR}$ includes the config file of student_model.
 
python general_distill.py --pregenerated_data ${CORPUS_JSON}$ \ 
                          --teacher_model ${BERT_BASE}$ \
                          --student_model ${STUDENT_CONFIG_DIR}$ \
                          --reduce_memory --do_lower_case \
                          --train_batch_size 256 \
                          --output_dir ${GENERAL_TINYBERT_DIR}$ 
```


We also provide the models of general TinyBERT here and users can skip the general distillation.

=================1st version to reproduce our results in the paper ===========================

[General_TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1dDigD7QBv1BmE6pWU71pFYPgovvEqOOj) 

[General_TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1wXWR00EHK-Eb7pbyw0VP234i2JTnjJ-x)

=================2nd version (2019/11/18) trained with more (book+wiki) and no `[MASK]` corpus =======

[General_TinyBERT_v2(4layer-312dim)](https://drive.google.com/open?id=1PhI73thKoLU2iliasJmlQXBav3v33-8z)

[General_TinyBERT_v2(6layer-768dim)](https://drive.google.com/open?id=1r2bmEsQe4jUBrzJknnNaBJQDgiRKmQjF)

=================Chinese version trained with WIKI and NEWS corpus =======

[General_TinyBERT_zh(4layer-312dim)](https://huggingface.co/huawei-noah/TinyBERT_4L_zh/tree/main)

[General_TinyBERT_zh(6layer-768dim)](https://huggingface.co/huawei-noah/TinyBERT_6L_zh/tree/main)

Data Augmentation
=================
Data augmentation aims to expand the task-specific training set. Learning more task-related examples, the generalization capabilities of student model can be further improved. We combine a pre-trained language model BERT and GloVe embeddings to do word-level replacement for data augmentation.

Use `data_augmentation.py` to run data augmentation and the augmented dataset `train_aug.tsv` is automatically saved into the corresponding ${GLUE_DIR/TASK_NAME}$
```

python data_augmentation.py --pretrained_bert_model ${BERT_BASE_DIR}$ \
                            --glove_embs ${GLOVE_EMB}$ \
                            --glue_dir ${GLUE_DIR}$ \  
                            --task_name ${TASK_NAME}$

```
Before running data augmentation of GLUE tasks you should download the [GLUE data](https://gluebenchmark.com/tasks) by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to some directory GLUE_DIR. And TASK_NAME can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE.

Task-specific Distillation
==========================
In the task-specific distillation, we re-perform the proposed Transformer distillation to further improve TinyBERT by focusing on learning the task-specific knowledge. 

Task-specific distillation includes two steps: (1) intermediate layer distillation; (2) prediction layer distillation.

Step 1: use `task_distill.py` to run the intermediate layer distillation.
```

# ${FT_BERT_BASE_DIR}$ contains the fine-tuned BERT-base model.

python task_distill.py --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${GENERAL_TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \ 
                       --output_dir ${TMP_TINYBERT_DIR}$ \
                       --max_seq_length 128 \
                       --train_batch_size 32 \
                       --num_train_epochs 10 \
                       --aug_train \
                       --do_lower_case  
                         
```


Step 2: use `task_distill.py` to run the prediction layer distillation.
```

python task_distill.py --pred_distill  \
                       --teacher_model ${FT_BERT_BASE_DIR}$ \
                       --student_model ${TMP_TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${TINYBERT_DIR}$ \
                       --aug_train  \  
                       --do_lower_case \
                       --learning_rate 3e-5  \
                       --num_train_epochs  3  \
                       --eval_step 100 \
                       --max_seq_length 128 \
                       --train_batch_size 32 
                       
```


We here also provide the distilled TinyBERT(both 4layer-312dim and 6layer-768dim) of all GLUE tasks for evaluation. Every task has its own folder where the corresponding model has been saved.

[TinyBERT(4layer-312dim)](https://drive.google.com/uc?export=download&id=1_sCARNCgOZZFiWTSgNbE7viW_G5vIXYg) 

[TinyBERT(6layer-768dim)](https://drive.google.com/uc?export=download&id=1Vf0ZnMhtZFUE0XoD3hTXc6QtHwKr_PwS)


Evaluation
==========================
The `task_distill.py` also provide the evalution by running the following command:

```
${TINYBERT_DIR}$ includes the config file, student model and vocab file.

python task_distill.py --do_eval \
                       --student_model ${TINYBERT_DIR}$ \
                       --data_dir ${TASK_DIR}$ \
                       --task_name ${TASK_NAME}$ \
                       --output_dir ${OUTPUT_DIR}$ \
                       --do_lower_case \
                       --eval_batch_size 32 \
                       --max_seq_length 128  
                                   
```

To Dos
=========================
* Evaluate TinyBERT on Chinese tasks.
* Tiny*: use NEZHA or ALBERT as the teacher in TinyBERT learning.
* Release better general TinyBERTs.
