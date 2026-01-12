import torch
import os

# Paths
input_path = "bert_base_uncased_teacher_RTE/pytorch_model_old.bin"
output_path = "bert_base_uncased_teacher_RTE/pytorch_model.bin"

print(f"Processing {input_path}...")
state_dict = torch.load(input_path, map_location="cpu")
new_state_dict = {}
found_classifier = False

for key, value in state_dict.items():
    new_key = key

    # Fix missing 'bert.' prefix for body layers
    if key.startswith(("encoder.", "embeddings.", "pooler.")):
        new_key = "bert." + key

    # Check for classifier head
    if "classifier" in key:
        found_classifier = True
        # Ensure exact match for classifier keys if needed
        if key in ["classifier.weight", "classifier.bias"]:
            new_key = key

    new_state_dict[new_key] = value

# Patch missing classifier if necessary
if not found_classifier:
    print("Warning: No classifier found. Initializing random head (accuracy will be random until trained).")
    new_state_dict["classifier.weight"] = torch.randn(2, 768) * 0.02
    new_state_dict["classifier.bias"] = torch.zeros(2)

torch.save(new_state_dict, output_path)
print(f"Done. Saved to {output_path}")