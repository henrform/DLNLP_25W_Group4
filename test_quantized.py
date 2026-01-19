from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoConfig
import torch

BITS = 8
OUTPUT_PATH = Path(f"./quantization/dynamic_{BITS}_bits_distilled_tinybert_6l_rte/")


# 1️⃣ Load config ONLY
config = AutoConfig.from_pretrained(OUTPUT_PATH)

model = AutoModelForSequenceClassification.from_config(config)

# 2️⃣ Load quantized weights manually
state_dict = torch.load(
    f"{OUTPUT_PATH}/pytorch_model.bin",
    map_location="cpu"
)

model.load_state_dict(state_dict)
model.eval()

# 3️⃣ Sanity check
print(type(model.bert.encoder.layer[0].attention.self.query))