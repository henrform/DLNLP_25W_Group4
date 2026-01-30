Reproduction
============

This is a clone of the repo provided by the original Authors of TinyBERT. You can find the original code [here](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) and the paper [here](https://arxiv.org/abs/1909.10351). We tried to preserve the original codebase as much as possible. Some additional scripts are necessary to make the original code work. We still had to modify some parts where the script crashed, most likely due to an old Python version. We also provide a script to run all these steps at once in `reproduction.sh`, but please not, that you still need to manually download the english wikipedia before. For more information about that, refer to the corresponding step below. Please also note that the script itself will run for roughly 7 days on an RTX 4090 just for the reproduction part. Extension will take about the same amount of time. All commands are tailored to that GPU. You might need to decrease the batch size if you have less VRAM available.

## Results
We got within a few percent of the reported scores in the paper, sometimes exceeding them and sometimes falling short. This is maybe due to a different number used for epochs and learning rate depending on the task. We used the values provided by the authors in the original repository.

| Model                                                | RTE      | SST-2    | CoLA     | MRPC     |
|------------------------------------------------------|----------|----------|----------|----------|
| TinyBERT_4 (paper)                                   | 66.6     | 92.6     | 44.1     | 86.4     |
| TinyBERT_4 (reproduction, batch size=64, lr=5e-5)    | 53.06    | 92.3     | 31.7     | 82.1     |
| TinyBERT_4 (reproduction, batch size=64, lr=1.25e-5) | 58.6     | 91.8     |  37.2    | 85.7     |
| TinyBERT_6 (paper)                                   | **70.0** | **93.1** | 51.1     | 87.3     |
| TinyBERT_6 (reproduction)                            | 69.0     | 92.3     | **54.3** | **88.8** |
| TinyBERT_6_smart_init_direct (no tsd)                | 47.3     | 83.1     | 28.2     | 17.7     |
| TinyBERT_6_smart_init_direct (tsd)                   | 67.5     | 92.0     | 55.7     | 88.0     |
| TinyBERT_6_smart_init                                | 69.3     | 93.3     | 56.2     | 88.5     |

## Setup and get required data

First, we need to create a conda environment and activate it
```bash
conda env create -f environment.yml
conda activate DLNLP_25W_Group4
```

Next, setup folder structure
```bash
mkdir -p data models
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
python general_distill.py --pregenerated_data json_wiki --teacher_model models/bert-base-uncased --student_model tinybert_4l_config --reduce_memory --do_lower_case --train_batch_size 64 --output_dir models/distilled_tinybert_4l --eval_step 10000
# With the reduced learning rate:
python general_distill.py --pregenerated_data json_wiki --teacher_model models/bert-base-uncased --student_model tinybert_4l_config --reduce_memory --do_lower_case --train_batch_size 64 --output_dir models/distilled_tinybert_4l --eval_step 10000   --learning_rate 1.25e-5
python general_distill.py --pregenerated_data json_wiki --teacher_model models/bert-base-uncased --student_model tinybert_6l_config --reduce_memory --do_lower_case --train_batch_size 256 --gradient_accumulation_steps 8 --output_dir models/distilled_tinybert_6l --eval_step 10000
```

We need to create a copy of the last
```bash
cp $(ls -t models/distilled_tinybert_4l/step_*_pytorch_model.bin | head -n1) models/distilled_tinybert_4l/pytorch_model.bin
cp $(ls -t models/distilled_tinybert_6l/step_*_pytorch_model.bin | head -n1) models/distilled_tinybert_6l/pytorch_model.bin
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
python finetune_teacher.py --data_dir data/glue_data/RTE --bert_model models/bert-base-uncased-rte --output_dir models/final_rte_teacher --num_train_epochs 5 --train_batch_size 32 --learning_rate 2e-5
python finetune_teacher.py --data_dir data/glue_data/SST-2 --bert_model models/bert-base-uncased-sst2 --output_dir models/final_sst2_teacher --num_train_epochs 5 --train_batch_size 32 --learning_rate 2e-5
python finetune_teacher.py --data_dir data/glue_data/CoLA --bert_model models/bert-base-uncased-cola --output_dir models/final_cola_teacher --num_train_epochs 5 --train_batch_size 32 --learning_rate 2e-5
python finetune_teacher.py --data_dir data/glue_data/MRPC --bert_model models/bert-base-uncased-mrpc --output_dir models/final_mrpc_teacher --num_train_epochs 5 --train_batch_size 32 --learning_rate 2e-5
```

The 12 commands below will run the task-specific distillation the 4 and 6 layer TinyBERT
```bash
# 4 layer task-specific distillation
python task_distill.py --teacher_model models/final_rte_teacher --student_model models/distilled_tinybert_4l --data_dir data/glue_data/RTE --task_name RTE --output_dir models/tmp_distilled_tinybert_4l_rte --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill --teacher_model models/final_rte_teacher --student_model models/tmp_distilled_tinybert_4l_rte --data_dir data/glue_data/RTE --task_name RTE --output_dir models/distilled_tinybert_4l_rte --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_cola_teacher --student_model models/distilled_tinybert_4l --data_dir data/glue_data/CoLA --task_name cola --output_dir models/tmp_distilled_tinybert_4l_cola --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_cola_teacher --student_model models/tmp_distilled_tinybert_4l_cola --data_dir data/glue_data/CoLA --task_name cola --output_dir models/distilled_tinybert_4l_cola --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_sst2_teacher --student_model models/distilled_tinybert_4l --data_dir data/glue_data/SST-2 --task_name sst-2 --output_dir models/tmp_distilled_tinybert_4l_sst2 --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_sst2_teacher --student_model models/tmp_distilled_tinybert_4l_sst2 --data_dir data/glue_data/SST-2 --task_name sst-2 --output_dir models/distilled_tinybert_4l_sst2 --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_mrpc_teacher --student_model models/distilled_tinybert_4l --data_dir data/glue_data/MRPC --task_name mrpc --output_dir models/tmp_distilled_tinybert_4l_mrpc --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_mrpc_teacher --student_model models/tmp_distilled_tinybert_4l_mrpc --data_dir data/glue_data/MRPC --task_name mrpc --output_dir models/distilled_tinybert_4l_mrpc --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32


# 6 layer task-specific distillation
python task_distill.py --teacher_model models/final_rte_teacher --student_model models/distilled_tinybert_6l --data_dir data/glue_data/RTE --task_name RTE --output_dir models/tmp_distilled_tinybert_6l_rte --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill --teacher_model models/final_rte_teacher --student_model models/tmp_distilled_tinybert_6l_rte --data_dir data/glue_data/RTE --task_name RTE --output_dir models/distilled_tinybert_6l_rte --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

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
python task_distill.py --do_eval --student_model models/distilled_tinybert_4l_sst2 --data_dir data/glue_data/SST-2 --task_name sst-2 --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/distilled_tinybert_4l_mrpc --data_dir data/glue_data/MRPC --task_name mrpc --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp

# 6 layer evaluation
python task_distill.py --do_eval --student_model models/distilled_tinybert_6l_rte --data_dir data/glue_data/RTE --task_name rte --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/distilled_tinybert_6l_cola --data_dir data/glue_data/CoLA --task_name cola --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/distilled_tinybert_6l_sst2 --data_dir data/glue_data/SST-2 --task_name sst-2 --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/distilled_tinybert_6l_mrpc --data_dir data/glue_data/MRPC --task_name mrpc --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
```

## Extensions
### Extending TinyBERT with K-Prune

This section contains our experimental extension of **TinyBERT** by integrating **K-Prune**, a structured pruning method, to further improve model efficiency while preserving performance. 

### Motivation

TinyBERT significantly reduces model size via knowledge distillation.  
In this work, we extend TinyBERT by applying **K-Prune** to investigate whether structured pruning can further compress distilled Transformer models without substantial accuracy degradation.

### Approach

- **Base model**: Pretrained TinyBERT
- **Extension**: K-Prune applied after distillation
- **Pruned components**:
  - Attention heads
  - Feed-forward network (FFN) neurons
- **Pruning strategy**:
  - Importance-based structured pruning
  - Fixed pruning ratio per layer

### References

 **K-Prune**  
  Seungcheol Park, Hojun Choi, U Kang *Accurate Retraining-free Pruning for Pretrained Encoder-based Language Models*  
  Paper: https://arxiv.org/pdf/2308.03449  
  Repository: https://github.com/snudm-starlab/K-prune/tree/main

### Key arguments of K-prune

- ckpt_dir: a path for the directory that contains fine-tuned checkpoints
- exp_name: the name of experiments for generating an output directory
- task_name: the name of a task to run, e.g. mrpc, qqp, sst2, stsb, mnli, qnli, squad, or squad_v2
- model_name: the name of a model to run, e.g. bert-base-uncased or distilbert-base-uncased
- gpu: the numbering of GPU to use
- seed: a random seed
- num_tokens: the number of tokens in a sample dataset
- constraint: a ratio of FLOPs to be removed
- lam_pred: a balance coefficient for predictive knowledge
- lam_rep: a balance coefficient for representational knowledge
- mu: a balance coefficient for importance scores of attention heads
- T: a temperature for softmax functions
- sublayerwise_tuning: whether perform layerwise tuning or not. Remove the argument if you do not use sub-layerwise tuning.

### Running
We suggest that you run it in a different environment since dependency issues might occur.You change to K-prune directory in order to run. The following code is an example of generating %10 compressed using K-PRUNE.
```bash
      python src/main.py \
    --model_name ${MODEL_DIR} \
    --task_name ${TASK_NAME} \
    --ckpt_dir ${MODEL_DIR} \
    --exp_name ${OUTPUT_DIR} \
    --gpu ${GPU} \
    --seed ${SEED} \
    --num_tokens 100000 \
    --constraint 0.1 \
    --lam_pred 1.0 \
    --lam_rep 2.5e-4 \
    --mu 64 \
    --T 2 \
    --sublayerwise_tuning
```



### Smart initialization for TinyBERT_6
The original paper used a random initialization for the student model. However, since the encoder layers of TinyBERT_6 have the same size as the ones of BERT, we can copy every second layer and glue those together. This way we get a model with initialized parameters. It's now possible to skip the general distillation entirely and get a model with similar performance to the one in the original paper. Using the smart initialization with both types of distillation presented in the paper yields slightly better scores.

This creates the base models from all the BERTs previously downloaded.
```bash
python tinybert_6_from_bert.py
```

The distillation now is the same as before.
```bash
# 6 layer smart-init direct task-specific distillation
python task_distill.py --teacher_model models/final_rte_teacher --student_model models/tinybert_6l_smart_init_direct_rte --data_dir data/glue_data/RTE --task_name RTE --output_dir models/tmp_tinybert_6l_smart_init_direct_rte --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill --teacher_model models/final_rte_teacher --student_model models/tmp_tinybert_6l_smart_init_direct_rte --data_dir data/glue_data/RTE --task_name RTE --output_dir models/tinybert_6l_smart_init_direct_rte_tsd --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_cola_teacher --student_model models/tinybert_6l_smart_init_direct_cola --data_dir data/glue_data/CoLA --task_name cola --output_dir models/tmp_tinybert_6l_smart_init_direct_cola --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_cola_teacher --student_model models/tmp_tinybert_6l_smart_init_direct_cola --data_dir data/glue_data/CoLA --task_name cola --output_dir models/tinybert_6l_smart_init_direct_cola_tsd --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_sst2_teacher --student_model models/tinybert_6l_smart_init_direct_sst2 --data_dir data/glue_data/SST-2 --task_name sst-2 --output_dir models/tmp_tinybert_6l_smart_init_direct_sst2 --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_sst2_teacher --student_model models/tmp_tinybert_6l_smart_init_direct_sst2 --data_dir data/glue_data/SST-2 --task_name sst-2 --output_dir models/tinybert_6l_smart_init_direct_sst2_tsd --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_mrpc_teacher --student_model models/tinybert_6l_smart_init_direct_mrpc --data_dir data/glue_data/MRPC --task_name mrpc --output_dir models/tmp_tinybert_6l_smart_init_direct_mrpc --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_mrpc_teacher --student_model models/tmp_tinybert_6l_smart_init_direct_mrpc --data_dir data/glue_data/MRPC --task_name mrpc --output_dir models/tinybert_6l_smart_init_direct_mrpc_tsd --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

# 6 layer smart-init direct evaluation
python task_distill.py --do_eval --student_model models/tinybert_6l_smart_init_direct_rte_tsd --data_dir data/glue_data/RTE --task_name rte --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/tinybert_6l_smart_init_direct_cola_tsd --data_dir data/glue_data/CoLA --task_name cola --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/tinybert_6l_smart_init_direct_sst2_tsd --data_dir data/glue_data/SST-2 --task_name sst-2 --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/tinybert_6l_smart_init_direct_mrpc_tsd --data_dir data/glue_data/MRPC --task_name mrpc --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp

# 6 layer smart-init general distillation
python general_distill.py --pregenerated_data data/json_wiki --teacher_model models/bert-base-uncased  --student_model models/tinybert_6l_smart_init_direct_base --reduce_memory --do_lower_case --train_batch_size 256 --gradient_accumulation_steps 8 --output_dir models/distilled_tinybert_6l_smart_init --eval_step 10000 --continue_train --num_train_epochs 1 --warmup_proportion 0.01

# 6 layer smart-init task-specific distillation
python task_distill.py --teacher_model models/final_rte_teacher --student_model models/distilled_tinybert_6l_smart_init --data_dir data/glue_data/RTE --task_name RTE --output_dir models/tmp_tinybert_6l_smart_init_rte --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill --teacher_model models/final_rte_teacher --student_model models/tmp_tinybert_6l_smart_init_rte --data_dir data/glue_data/RTE --task_name RTE --output_dir models/tinybert_6l_smart_init_rte --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_cola_teacher --student_model models/distilled_tinybert_6l_smart_init --data_dir data/glue_data/CoLA --task_name cola --output_dir models/tmp_tinybert_6l_smart_init_cola --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_cola_teacher --student_model models/tmp_tinybert_6l_smart_init_cola --data_dir data/glue_data/CoLA --task_name cola --output_dir models/tinybert_6l_smart_init_cola --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_sst2_teacher --student_model models/distilled_tinybert_6l_smart_init --data_dir data/glue_data/SST-2 --task_name sst-2 --output_dir models/tmp_tinybert_6l_smart_init_sst2 --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_sst2_teacher --student_model models/tmp_tinybert_6l_smart_init_sst2 --data_dir data/glue_data/SST-2 --task_name sst-2 --output_dir models/tinybert_6l_smart_init_sst2 --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

python task_distill.py --teacher_model models/final_mrpc_teacher --student_model models/distilled_tinybert_6l_smart_init --data_dir data/glue_data/MRPC --task_name mrpc --output_dir models/tmp_tinybert_6l_smart_init_mrpc --max_seq_length 128 --train_batch_size 32 --num_train_epochs 10 --aug_train --do_lower_case
python task_distill.py --pred_distill  --teacher_model models/final_mrpc_teacher --student_model models/tmp_tinybert_6l_smart_init_mrpc --data_dir data/glue_data/MRPC --task_name mrpc --output_dir models/tinybert_6l_smart_init_mrpc --aug_train  --do_lower_case --learning_rate 3e-5  --num_train_epochs  3  --eval_step 100 --max_seq_length 128 --train_batch_size 32

# 6 layer smart-init evaluation
python task_distill.py --do_eval --student_model models/tinybert_6l_smart_init_rte --data_dir data/glue_data/RTE --task_name rte --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/tinybert_6l_smart_init_cola --data_dir data/glue_data/CoLA --task_name cola --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/tinybert_6l_smart_init_sst2 --data_dir data/glue_data/SST-2 --task_name sst-2 --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
python task_distill.py --do_eval --student_model models/tinybert_6l_smart_init_mrpc --data_dir data/glue_data/MRPC --task_name mrpc --do_lower_case --eval_batch_size 32 --max_seq_length 128 --output_dir tmp
```

### Quantization

In addition to pruning and smart initialization, *post-training quantization* of
a task-distilled TinyBERT model was attempted (unsuccessfully). The code
regarding quantization can be found on the corresponding `quantization` branch
which was not merged into `main`.

#### References

The following paper describes quantization of the TinyBERT model updating weight values to 16 bit floating point numbers, however the resulting model was not evaluated using the *GLUE* benchmark:

> [TYMBERT: Tiny Yet Mighty– Fine-Tuned and Compressed TinyBERT for High-Accuracy, Low-Resource NLP](https://ieeexplore.ieee.org/abstract/document/11160230)
