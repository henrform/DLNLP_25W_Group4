import torch
from nlp_architect.models.transformers.quantized_bert import QuantizedBertModel
from transformers import BertTokenizer, BertConfig

# 1. Load your Task-Distilled TinyBERT 
# (Assuming you have already distilled it into a smaller config)
model_path = "path/to/distilled/tinybert"
config = BertConfig.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# 2. Initialize the Quantized version of the model
# This automatically replaces Linear/Embedding layers with 8-bit simulated ones
quantized_tinybert = QuantizedBertModel.from_pretrained(
    model_path, 
    config=config
)

# 3. Setup Training (Standard PyTorch/HuggingFace style)
# The key is that 'quantized_tinybert' now contains observers 
# that track the min/max ranges of activations during fine-tuning.
optimizer = torch.optim.AdamW(quantized_tinybert.parameters(), lr=2e-5)

# Example Training Loop Snippet
quantized_tinybert.train()
for batch in train_dataloader:
    inputs = {k: v.to(device) for k, v in batch.items()}
    outputs = quantized_tinybert(**inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# 4. Save for Inference
# NLP Architect saves two versions: a 'fake' quantized FP32 model 
# and the actual INT8 weights.
quantized_tinybert.save_pretrained("./output/quantized_tinybert")