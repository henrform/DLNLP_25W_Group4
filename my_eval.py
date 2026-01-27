import torch
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate

# --- CONFIGURATION ---
TASK_NAME = "rte"  # Change to "sst2", "mrpc", etc.
# MODEL_DIR = Path("./quantization/distilled_tinybert_6l_rte/")
MODEL_DIR = Path("./quantization/dynamic_8_bits_distilled_tinybert_6l_rte/")
DATA_DIR = "./quantization/glue_data/RTE"  # Path to your local GLUE folder


def load_quantized_model(checkpoint_path, num_labels):
    config = AutoConfig.from_pretrained(checkpoint_path)
    # We use AutoModelForSequenceClassification to get the .classifier head
    model = AutoModelForSequenceClassification.from_config(config)
    
    # Apply dynamic quantization to match the saved structure
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    # Load weights
    state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
    quantized_model.load_state_dict(state_dict)
    quantized_model.eval()
    return quantized_model


def evaluate_glue():
    # 1. Load Metric and Tokenizer
    metric = evaluate.load("glue", TASK_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    # 2. Load Local Data
    # Most GLUE tasks have 'sentence1', 'sentence2' or just 'sentence'
    # We load from local files (tsv) usually found in GLUE folders
    dataset = load_dataset("csv", data_files={
        "validation": os.path.join(DATA_DIR, "dev.tsv")
    }, delimiter="\t", quoting=3)
    
    # 3. Load Quantized Model
    num_labels = 2 if TASK_NAME != "mnli" else 3
    model = load_quantized_model(MODEL_DIR, num_labels)
    
    # 4. Task Mapping (Extensibility)
    task_to_keys = {
        "rte": ("sentence1", "sentence2"),
        "sst2": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qnli": ("question", "sentence"),
    }
    label_map = {"entailment": 0, "not_entailment": 1}
    sentence1_key, sentence2_key = task_to_keys[TASK_NAME]

    # 5. Evaluation Loop
    print(f"Evaluating {TASK_NAME}...")
    for example in tqdm(dataset["validation"]):
        # Prepare inputs
        texts = (example[sentence1_key],) if sentence2_key is None else (example[sentence1_key], example[sentence2_key])
        inputs = tokenizer(*texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)

        raw_label = example["label"]
        label_int = label_map[raw_label] if isinstance(raw_label, str) else raw_label
        
        metric.add(predictions=predictions, references=label_int)

    # 6. Final Score
    results = metric.compute()
    print(f"\nResults for {TASK_NAME}: {results}")


if __name__ == "__main__":
    evaluate_glue()