from transformers import BertModel, BertTokenizer

output_dir = "bert_base_uncased_teacher"

model = BertModel.from_pretrained("bert-base-uncased")
model.save_pretrained(output_dir, safe_serialization=False)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer.save_pretrained(output_dir)