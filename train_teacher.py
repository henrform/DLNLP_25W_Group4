# Example command:
# python train_teacher.py --data_dir glue_data/RTE --bert_model bert_base_uncased_teacher_RTE --output_dir final_rte_teacher --num_train_epochs 5 --train_batch_size 32 --learning_rate 2e-5
import os
import csv
import argparse
import torch
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from transformer.modeling import TinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer
from transformer.optimization import BertAdam


class InputExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for ex in examples:
        tokens_a = tokenizer.tokenize(ex.text_a)
        tokens_b = tokenizer.tokenize(ex.text_b)
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        label_id = label_map[ex.label]
        features.append(InputFeatures(input_ids, input_mask, segment_ids, label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_length: break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--bert_model", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=32, type=int)
    parser.add_argument("--num_train_epochs", default=3.0, type=float)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_examples = []
    with open(os.path.join(args.data_dir, "train.tsv"), "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=None)
        for i, line in enumerate(reader):
            if i == 0: continue
            if len(line) < 4:
                continue
            train_examples.append(InputExample(guid=i, text_a=line[1], text_b=line[2], label=line[3]))

    label_list = ["entailment", "not_entailment"]
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=True)
    model = TinyBertForSequenceClassification.from_pretrained(args.bert_model, num_labels=2)
    model.to(device)

    train_features = convert_examples_to_features(train_examples, label_list, args.max_seq_length, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    param_optimizer = list(model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if 'bias' not in n], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if 'bias' in n], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters, lr=args.learning_rate, warmup=0.1,
                         t_total=int(len(train_dataloader) * args.num_train_epochs))
    loss_fct = CrossEntropyLoss()

    model.train()
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            logits, _, _ = model(input_ids, segment_ids, input_mask)
            loss = loss_fct(logits.view(-1, 2), label_ids.view(-1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    model_to_save.config.to_json_file(os.path.join(args.output_dir, "config.json"))
    tokenizer.save_vocabulary(args.output_dir)


if __name__ == "__main__":
    main()