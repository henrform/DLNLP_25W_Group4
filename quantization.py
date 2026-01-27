from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPTQConfig, AutoModel

BITS = 8
MODEL_PATH = "./quantization/distilled_tinybert_6l_rte/"
OUTPUT_PATH = f"./quantization/quantized_{BITS}_bits_distilled_tinybert_6l_rte/"


def quantize_model(model_path: str, output_path: str, bits: int) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    gptq_config = GPTQConfig(
        bits=bits,
        tokenizer=tokenizer,
        use_exllama=False,
        use_cuda_fp16=True,
        dataset="c4-new",
        block_name_to_quantize="encoder.layer"
    )

    print(f"Starting quantization for local model at: '{model_path}'")
    quantized_model = AutoModel.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=gptq_config
    )

    print(quantized_model)

    quantized_model.save_pretrained(output_path, safe_serialization=False)
    tokenizer.save_pretrained(output_path)

    print(f"Quantization complete. Saved to: '{output_path}'")


def main():
    quantize_model(MODEL_PATH, OUTPUT_PATH, BITS)


if __name__ == "__main__":
    main()