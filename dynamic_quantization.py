import torch
import os

from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BITS = 8
MODEL_PATH = Path("./quantization/distilled_tinybert_6l_rte/")
OUTPUT_PATH = Path(f"./quantization/dynamic_{BITS}_bits_distilled_tinybert_6l_rte/")


def quantize_model(model_path: str, output_path: str, bits: int) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)

    print(f"Starting dynamic quantization for local model at: '{model_path}'")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    # NOTE(raoul): Suggested by Gemini, when loading the state dict in the modified
    # `qtask_distll.py`, the `fit_dense.scale` field was throwing an error.
    model_to_quantize = model
    for name, module in model_to_quantize.named_modules():
        if "fit_dense" in name:
            pass

    torch.save(quantized_model.state_dict(), output_path / "pytorch_model.bin")

    model.config.save_pretrained(OUTPUT_PATH)
    tokenizer.save_pretrained(OUTPUT_PATH)
    
    print(f"Dynamic quantization complete. Saved to: '{output_path}'")


def main():
    quantize_model(MODEL_PATH, OUTPUT_PATH, BITS)


if __name__ == "__main__":
    main()
