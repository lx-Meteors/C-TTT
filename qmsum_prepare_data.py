import os
import json
import random
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--work_dir',
        type=str,
        required=False,
        default="../experiment/vanilla/qmsum",
        help='Directory including the configuration file'
    )
    return parser.parse_args()


def _dataset_name_from_path(instruction_dataset_repo: str) -> str:
    return os.path.splitext(os.path.basename(instruction_dataset_repo))[0]


def get_examples_list(instruction_dataset_repo, split):
    dataset_name = _dataset_name_from_path(instruction_dataset_repo)
    cache_path = f'output/{dataset_name}_{split}_instruction_dataset.json'

    if os.path.exists(cache_path):
        with open(cache_path, 'r', encoding='utf-8') as f:
            examples_list = json.load(f)
        return examples_list

    examples_list = []
    with open(instruction_dataset_repo, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Loading {dataset_name} {split} examples"):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)

            answers = raw.get("answers", [])
            if not isinstance(answers, list):
                answers = [str(answers)]
            answers = [a for a in answers if isinstance(a, str) and a.strip()]
            if len(answers) == 0:
                continue

            example = {
                "input": raw["input"],          # QMSum query
                "context": raw["context"],      # meeting transcript
                "answers": answers,              # multi-reference summaries
                "answer": answers[0],            # keep first ref for convenience/debugging
                "length": raw.get("length"),
                "dataset": raw.get("dataset"),
                "language": raw.get("language"),
                "all_classes": raw.get("all_classes"),
                "_id": raw.get("_id"),
            }
            examples_list.append(example)

    with open(cache_path, 'w', encoding='utf-8') as f:
        json.dump(examples_list, f, ensure_ascii=False)

    return examples_list


def get_ids(instruction_dataset_repo_name, examples_list, tokenizer, split):
    examples = []
    minn = 999999
    maxn = 0

    for example in tqdm(examples_list, desc=f"Tokenizing {split} examples"):
        if split == "test":
            # QMSum test-time inference: use real context + real query, keep refs only in cached raw file.
            context = tokenizer(example["context"], add_special_tokens=False)["input_ids"]
            prompt = tokenizer(example["input"], add_special_tokens=False)["input_ids"]
            answer = tokenizer(example["answers"][0], add_special_tokens=False)["input_ids"]

            context_ids = [tokenizer.bos_token_id] + tokenizer("### Context:\n", add_special_tokens=False)["input_ids"] + context
            question_ids = tokenizer("\n### Question:\n", add_special_tokens=False)["input_ids"] + prompt \
                           + tokenizer("\n### Answer:\n", add_special_tokens=False)["input_ids"]
            answer_ids = answer + [tokenizer.eos_token_id]

            tot_len = len(context_ids) + len(question_ids) + len(answer_ids)
            minn = min(minn, tot_len)
            maxn = max(maxn, tot_len)

            inputs = torch.LongTensor(context_ids)
            lm_target = torch.LongTensor(question_ids)
            examples.append({"input_ids": inputs, "lm_targets": lm_target})
        else:
            # Keep exactly the PwC-style TTT data construction:
            # split the in-domain/OOD document itself into context/target continuation.
            context_temp = tokenizer(example["context"], add_special_tokens=False)["input_ids"]
            if len(context_temp) < 2:
                continue

            context = context_temp[:len(context_temp) // 2]
            answer = context_temp[len(context_temp) // 2:]
            prompt = tokenizer("Predict the following text from the given context.", add_special_tokens=False)["input_ids"]

            context_ids = [tokenizer.bos_token_id] + tokenizer("### Context:\n", add_special_tokens=False)["input_ids"] + context
            question_ids = tokenizer("\n### Question:\n", add_special_tokens=False)["input_ids"] + prompt \
                           + tokenizer("\n### Answer:\n", add_special_tokens=False)["input_ids"]
            answer_ids = answer + [tokenizer.eos_token_id]

            tot_len = len(context_ids) + len(question_ids) + len(answer_ids)
            minn = min(minn, tot_len)
            maxn = max(maxn, tot_len)

            instruction_target = [-100 for _ in question_ids] + [x for x in answer_ids]
            instruction_target = instruction_target[1:]

            inputs = torch.LongTensor(context_ids)
            lm_target = torch.LongTensor(question_ids + answer_ids)
            instruction_target = torch.LongTensor(instruction_target)
            examples.append({
                "input_ids": inputs,
                "lm_targets": lm_target,
                "instruction_target": instruction_target
            })

    print(f"len range: [{minn}:{maxn}]")
    return examples


def get_examples(model_id, instruction_dataset_repo, samples_num, min_len, max_len, dataset_repo):
    model_name = model_id.split('/')[-1]
    instruction_dataset_repo_name = _dataset_name_from_path(instruction_dataset_repo)
    train_data_name = f"output/{instruction_dataset_repo_name}_train_{model_name}_{samples_num}samples_instruction.pt"
    eval_data_name = f"output/{instruction_dataset_repo_name}_eval_{model_name}_{samples_num}samples_instruction.pt"

    print(f"in:train_data_name:{train_data_name}")
    if os.path.exists(train_data_name) and os.path.exists(eval_data_name):
        print("loading data...")
        return torch.load(train_data_name), torch.load(eval_data_name)

    print(f"preparing data :train_data_name:{train_data_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Keep the same design as PwC: TTT train data comes from the same target-domain set.
    test_examples_list = get_examples_list(instruction_dataset_repo, split="test")
    train_examples_list = test_examples_list

    random.seed(0)
    train_data = get_ids(instruction_dataset_repo_name, train_examples_list, tokenizer, split="train")
    test_data = get_ids(instruction_dataset_repo_name, test_examples_list, tokenizer, split="test")

    torch.save(train_data, train_data_name)
    torch.save(test_data, eval_data_name)

    return train_data, test_data


if __name__ == "__main__":
    args = parse_args()
    with open(args.work_dir + "/config.json") as f:
        config = json.load(f)

    if not os.path.exists("output"):
        os.makedirs("output")

    training_config = config["sft_training_config"]
    config["data_config"]["model_id"] = training_config["model_id"]

    print(config["data_config"])
    train_examples, eval_examples = get_examples(**config["data_config"])
    print(len(train_examples))
    print(train_examples[50])
    print(len(eval_examples))
    print(eval_examples[50])
