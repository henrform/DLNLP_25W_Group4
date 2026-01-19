import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPTQConfig
from datasets import load_dataset

# NOTE(raoul): No difference from `quantization.py` in results.

BITS = 8
MODEL_PATH = "./quantization/distilled_tinybert_6l_rte/"
OUTPUT_PATH = f"./quantization/quantized_{BITS}_bits_distilled_tinybert_6l_rte/"

def get_rte_calibration_data(tokenizer, n_samples=128):
    """Loads a small sample of RTE data to guide the quantization math."""
    dataset = load_dataset("glue", "rte", split="train", trust_remote_code=True)
    samples = []
    for i in range(min(len(dataset), n_samples)):
        # Combine premise and hypothesis as the model would see it during inference
        text = f"{dataset[i]['sentence1']} {dataset[i]['sentence2']}"
        samples.append(text)
    return samples

def quantize_model(model_path: str, output_path: str, bits: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # FIX 1: Create local calibration data
    calibration_data = get_rte_calibration_data(tokenizer)

    # FIX 2: Update GPTQConfig to use the local samples
    gptq_config = GPTQConfig(
        bits=bits,
        tokenizer=tokenizer,
        use_exllama=False,
        use_cuda_fp16=True,
        dataset=calibration_data, # Use local data instead of "c4-new"
        block_name_to_quantize="bert.encoder.layer",
        # desc_act=True, # Improves accuracy for small models by quantizing important weights first
        # sym=True       # Symmetric quantization is generally more stable for BERT
    )

    print(f"Starting quantization with task-specific calibration...")
    quantized_model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=gptq_config
    )

    quantized_model.save_pretrained(output_path, safe_serialization=False)
    tokenizer.save_pretrained(output_path)
    print(f"Success! Saved to: '{output_path}'")

if __name__ == "__main__":
    quantize_model(MODEL_PATH, OUTPUT_PATH, BITS)