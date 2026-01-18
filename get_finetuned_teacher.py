import os
from transformers import BertModel, BertTokenizer

models_to_download = [
    "JeremiahZ/bert-base-uncased-qnli",
    "JeremiahZ/bert-base-uncased-sst2",
    "JeremiahZ/bert-base-uncased-cola",
    "JeremiahZ/bert-base-uncased-rte"
]

base_output_dir = "models"

for model_name in models_to_download:
    short_name = model_name.split('/')[-1]
    output_dir = os.path.join(base_output_dir, short_name)

    model = BertModel.from_pretrained(model_name)
    model.save_pretrained(output_dir, safe_serialization=False)

    tokenizer = BertTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)