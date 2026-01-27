import torch
import os
import evaluate
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from datasets import load_dataset

# --- CONFIGURATION ---
TASK_NAME = "rte"
MODEL_DIR = Path("./quantization/dynamic_8_bits_distilled_tinybert_6l_rte/")
# Path or Hugging Face ID for the original unquantized model
ORIGINAL_MODEL_ID = "./quantization/distilled_tinybert_6l_rte/" # Update this to your specific fine-tuned path
DATA_DIR = "./quantization/glue_data/RTE"

def load_quantized_model(checkpoint_path):
    config = AutoConfig.from_pretrained(checkpoint_path)
    model = AutoModelForSequenceClassification.from_config(config)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    state_dict = torch.load(checkpoint_path / "pytorch_model.bin", map_location="cpu")
    quantized_model.load_state_dict(state_dict)
    quantized_model.eval()
    return quantized_model

def load_original_model(model_id_or_path, num_labels):
    """Loads the non-quantized, full-precision model."""
    print(f"Loading original model from {model_id_or_path}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id_or_path, 
        num_labels=num_labels
    )
    model.eval()
    return model

def run_evaluation(model, dataset, tokenizer, metric, task_keys, label_map):
    sentence1_key, sentence2_key = task_keys
    for example in tqdm(dataset):
        texts = (example[sentence1_key],) if sentence2_key is None else (example[sentence1_key], example[sentence2_key])
        inputs = tokenizer(*texts, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        raw_label = example["label"]
        label_int = label_map[raw_label] if isinstance(raw_label, str) else raw_label
        metric.add(prediction=predictions.item(), reference=label_int)
    
    return metric.compute()

def evaluate_glue():
    metric = evaluate.load("glue", TASK_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    
    dataset = load_dataset("csv", data_files={"validation": os.path.join(DATA_DIR, "dev.tsv")}, delimiter="\t", quoting=3)
    num_labels = 2 if TASK_NAME != "mnli" else 3
    
    task_to_keys = {"rte": ("sentence1", "sentence2"), "sst2": ("sentence", None)}
    label_map = {"entailment": 0, "not_entailment": 1}

    # 1. Evaluate Quantized Model
    print("\n--- Evaluating Quantized Model ---")
    q_model = load_quantized_model(MODEL_DIR)
    q_results = run_evaluation(q_model, dataset["validation"], tokenizer, metric, task_to_keys[TASK_NAME], label_map)
    print(f"Quantized Results: {q_results}")

    # 2. Evaluate Original Model
    print("\n--- Evaluating Original Model ---")
    orig_model = load_original_model(ORIGINAL_MODEL_ID, num_labels)
    orig_results = run_evaluation(orig_model, dataset["validation"], tokenizer, metric, task_to_keys[TASK_NAME], label_map)
    print(f"Original Results: {orig_results}")

if __name__ == "__main__":
    evaluate_glue()