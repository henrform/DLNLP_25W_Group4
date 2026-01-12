import os

from transformers import AutoModelForMaskedLM, AutoTokenizer

BERT_BASE_UNCASED_ID = "google-bert/bert-base-uncased"
TINYBERT_GENERAL_4L_ID = "huawei-noah/TinyBERT_General_4L_312D"
TINYBERT_GENERAL_6L_ID = "huawei-noah/TinyBERT_General_6L_768D"

BERT_BASE_UNCASED_SAVE_DIR = "bert_base_uncased/"
TINYBERT_GENERAL_4L_SAVE_DIR = "tinybert_general_4l/"
TINYBERT_GENERAL_6L_SAVE_DIR = "tinybert_general_6l/"


def download_model(model_id: str, save_dir: str, overwrite_existing: bool = False) -> None:
    print(f"Downloading model {model_id}...")

    model = AutoModelForMaskedLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if os.path.exists(save_dir) and not overwrite_existing:
        return print(f"{save_dir} already exists, skipping {model_id}.")

    os.makedirs(save_dir, exist_ok=True)

    model.save_pretrained(save_dir, safe_serialization=False)
    tokenizer.save_pretrained(save_dir)

    print(f"Model and Tokenizer saved to {save_dir}.")


def main() -> None:
    download_model(BERT_BASE_UNCASED_ID, BERT_BASE_UNCASED_SAVE_DIR)
    download_model(TINYBERT_GENERAL_4L_ID, TINYBERT_GENERAL_4L_SAVE_DIR)
    download_model(TINYBERT_GENERAL_6L_ID, TINYBERT_GENERAL_6L_SAVE_DIR)


if __name__ == "__main__":
    main()