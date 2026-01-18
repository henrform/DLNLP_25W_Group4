import torch
import os

model_folders = [
    "bert-base-uncased-qnli",
    "bert-base-uncased-sst2",
    "bert-base-uncased-cola",
    "bert-base-uncased-rte"
]

base_dir = "models"

for folder in model_folders:
    model_path = os.path.join(base_dir, folder, "pytorch_model.bin")

    if not os.path.exists(model_path):
        print(f"Skipping {folder}: {model_path} not found.")
        continue

    print(f"Fixing layer names for {folder}...")
    state_dict = torch.load(model_path, map_location="cpu")
    new_state_dict = {}
    found_classifier = False

    for key, value in state_dict.items():
        new_key = key

        if key.startswith(("encoder.", "embeddings.", "pooler.")):
            new_key = "bert." + key

        if "classifier" in key:
            found_classifier = True

        new_state_dict[new_key] = value

    if not found_classifier:
        print(f"  Note: No classifier found in {folder}. Initializing a default head.")
        new_state_dict["classifier.weight"] = torch.randn(2, 768) * 0.02
        new_state_dict["classifier.bias"] = torch.zeros(2)

    torch.save(new_state_dict, model_path)