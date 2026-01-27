import torch
import os

from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer


BITS = 8
MODEL_PATH = Path("./quantization/distilled_tinybert_6l_rte/")
OUTPUT_PATH = Path(f"./quantization/dynamic_{BITS}_bits_distilled_tinybert_6l_rte/")


def quantize_model(model_path: str, output_path: str, bits: int) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)    
    # model = AutoModel.from_pretrained(model_path)

    
    # tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    # model = TinyBertForSequenceClassification.from_pretrained(model_path, num_labels=2)

    print(f"Starting dynamic quantization for local model at: '{model_path}'")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )

    print(quantized_model)

    torch.save(quantized_model.state_dict(), output_path / "pytorch_model.bin")

    tokenizer.save_pretrained(output_path)
    model.config.save_pretrained(output_path)
    
    # tokenizer.save_vocabulary(output_path)
    # model.config.to_json_file(output_path / "config.json")
    
    print(f"Dynamic quantization complete. Saved to: '{output_path}'")


def main():
    quantize_model(MODEL_PATH, OUTPUT_PATH, BITS)


if __name__ == "__main__":
    main()
