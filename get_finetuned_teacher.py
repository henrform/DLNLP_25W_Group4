from transformers import BertModel, BertTokenizer

output_dir = "bert_base_uncased_teacher_RTE"

model = BertModel.from_pretrained("JeremiahZ/bert-base-uncased-rte")
model.save_pretrained(output_dir, safe_serialization=False)

tokenizer = BertTokenizer.from_pretrained("JeremiahZ/bert-base-uncased-rte")
tokenizer.save_pretrained(output_dir)