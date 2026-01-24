from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
import torch, os, copy

models_to_process = [
    "final_mrpc_teacher",
    "final_sst2_teacher",
    "final_cola_teacher",
    "final_rte_teacher",
    "bert_base_uncased"
]

keep = [0, 2, 4, 6, 8, 10]

for model_name in models_to_process:
    teacher_path = os.path.join("models", model_name)
    task_name = model_name.split("_")[1]
    output_dir = os.path.join("models", f"tinybert_6l_init_from_bert_{task_name}")

    print(f"Processing {teacher_path} -> {output_dir}")

    # Load Teacher
    model = TinyBertForSequenceClassification.from_pretrained(teacher_path, num_labels=2)
    tokenizer = BertTokenizer.from_pretrained(teacher_path)

    # Strip Layers
    state = model.state_dict()
    new_state = {}

    for k, v in state.items():
        if "bert.encoder.layer" in k:
            idx = int(k.split(".")[3])
            if idx in keep:
                new_k = k.replace(f"layer.{idx}.", f"layer.{keep.index(idx)}.")
                new_state[new_k] = v
        else:
            new_state[k] = v

    # Save
    config = copy.deepcopy(model.config)
    config.num_hidden_layers = 6
    os.makedirs(output_dir, exist_ok=True)

    torch.save(new_state, os.path.join(output_dir, "pytorch_model.bin"))
    config.to_json_file(os.path.join(output_dir, "config.json"))
    tokenizer.save_vocabulary(output_dir)