# Create conda env
conda env create -f environment.yml
conda activate DLNLP_25W_Group4

# Create folders
mkdir -p data models

# Get GloVe data
wget https://nlp.stanford.edu/data/glove.840B.300d.zip
mv glove.840B.300d.zip data/
unzip data/glove.840B.300d.zip

# Get GLUE data
python download_glue.py
mv glue_data data/

# Extract wiki. This is the only step that requires manual intervention, by downloading the data
# from https://meta.wikimedia.org/wiki/Data_dump_torrents#English_Wikipedia and placing it in data/
python -m wikiextractor.WikiExtractor data/enwiki-20251101-pages-articles-multistream.xml.bz2 --json -o data/extracted_wiki

# Format wiki data for pregenerate_training_data.py
python prepare_data.py

# Download all necessary BERT models
python get_bert_models.py

# Pregenerate training data
python pregenerate_training_data.py --train_corpus wiki_for_bert.txt --bert_model models/bert-base-uncased --reduce_memory --do_lower_case --epochs_to_generate 3 --output_dir data/json_wiki

# General distillation
python general_distill.py --pregenerated_data json_wiki --teacher_model models/bert-base-uncased --student_model tinybert_4l_config --reduce_memory --do_lower_case --train_batch_size 256 --gradient_accumulation_steps 8 --output_dir models/distilled_tinybert_4l --eval_step 10000
python general_distill.py --pregenerated_data json_wiki --teacher_model models/bert-base-uncased --student_model tinybert_6l_config --reduce_memory --do_lower_case --train_batch_size 256 --gradient_accumulation_steps 8 --output_dir models/distilled_tinybert_6l --eval_step 10000

# Data augmentation
python data_augmentation.py --pretrained_bert_model models/bert-base-uncased --glove_embs data/glove.840B.300d.txt --glue_dir data/glue_data --task_name RTE
python data_augmentation.py --pretrained_bert_model models/bert-base-uncased --glove_embs data/glove.840B.300d.txt --glue_dir data/glue_data --task_name SST-2
python data_augmentation.py --pretrained_bert_model models/bert-base-uncased --glove_embs data/glove.840B.300d.txt --glue_dir data/glue_data --task_name CoLA
python data_augmentation.py --pretrained_bert_model models/bert-base-uncased --glove_embs data/glove.840B.300d.txt --glue_dir data/glue_data --task_name MRPC

# Fix downloaded fine-tuned BERT teachers
python fix_teacher_layers.py
python finetune_teacher.py

# Run task-specific distillation
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

# Evaluation
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