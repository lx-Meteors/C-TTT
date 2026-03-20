import json
import os
import sys
import logging
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import BASE_PATH

sys.path.append(BASE_PATH)
from qmsum_prepare_data import get_examples
from model.modeling import get_model, load_adapter
from instruction_dataloader import get_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str, default='../experiment/vanilla/qmsum', required=False,
                        help='Directory including the configuration file')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='total batch size')
    return parser.parse_args()


class Evaluator:
    def __init__(self, config, work_dir, batch_size, tokenizer):
        self.config = config
        self.work_dir = work_dir
        self.batch_size = batch_size
        self.device_count = torch.cuda.device_count()
        self.tokenizer = tokenizer

    def draw_ema_loss(self, alpha):
        def exponential_moving_average(values, alpha):
            ema = [values[0]]
            for value in values[1:]:
                ema.append(alpha * value + (1 - alpha) * ema[-1])
            return ema

        with open(os.path.join(self.work_dir, "instruction_info.json")) as f:
            info_list = json.load(f)

        lm_loss_values = [-1 if 'lm_loss' not in entry['training_loss'] else entry['training_loss']["lm_loss"] for entry in info_list]
        step_values = [entry['steps'] for entry in info_list]

        plt.figure(figsize=(10, 5))
        if lm_loss_values and lm_loss_values[0] != -1:
            plt.plot(step_values, exponential_moving_average(lm_loss_values, alpha=alpha), label="lm_loss")

        plt.xlabel("step")
        plt.ylabel(f"loss(ema_alpha={alpha})")
        plt.title(f"{self.work_dir}_loss(ema_alpha={alpha})")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.savefig(os.path.join(self.work_dir, 'instruction_ema_training_loss.png'))
        plt.close()

    def evaluate(self, rank=0):
        training_config = self.config["sft_training_config"]
        task_config = self.config["sft_task_config"]
        _, eval_examples = get_examples(**self.config["data_config"])
        example_num_per_gpu = len(eval_examples) // training_config["device_count"]

        if rank <= self.device_count - 2:
            eval_examples = eval_examples[rank * example_num_per_gpu:(rank + 1) * example_num_per_gpu]
        else:
            eval_examples = eval_examples[rank * example_num_per_gpu:]

        print(f"[INFO] GPU{rank}: eval_examples[{rank * example_num_per_gpu}:{rank * example_num_per_gpu + len(eval_examples)}], nums:{len(eval_examples)}")

        dataset = get_dataset(task_config["task_type"], eval_examples, batch_size=self.batch_size)
        loader = DataLoader(dataset, batch_size=None)

        model = get_model(training_config["model_id"], task_config, rank)
        model = load_adapter(model, save_path_and_name='/mnt/zhaorunsong/lx/SAC-TTT/experiment/checkpoint/vanilla/output/instruction_adapter.pt', log=True)  # 修改出checkpointd pt
        model.eval()

        info_list = []
        with torch.no_grad():
            for inputs in tqdm(loader, total=len(eval_examples) // self.batch_size):
                inputs = {key: (value.to(rank) if value is not None else None) for key, value in inputs.items()}
                generate_text = model.lm_inference(inputs)
                info_list.append({"generate_text": generate_text})

        with open(self.work_dir + f'/instruction_eval_info_list_{rank}.json', 'w', encoding='utf-8') as f:
            json.dump(info_list, f, ensure_ascii=False)

    def run(self, rank):
        if rank == 0:
            self.draw_ema_loss(alpha=0.1)
        self.evaluate(rank)


def evaluate(rank, args, world_size, tokenizer):
    with open(args.work_dir + "/output/config.json") as f:
        config = json.load(f)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(args.work_dir + f'/output/instruction_evaluate_info_rank{rank}.txt', mode='w'),
            logging.StreamHandler()
        ]
    )

    evaluator = Evaluator(config, args.work_dir + "/output", args.batch_size, tokenizer)
    evaluator.run(rank)


if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()

    with open(args.work_dir + "/output/config.json") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["data_config"]["model_id"])

    sys.setrecursionlimit(10000)
    if not os.path.exists(args.work_dir + '/output/instruction_eval_info_list_0.json'):
        mp.spawn(evaluate, args=(args, world_size, tokenizer), nprocs=world_size, join=True)

    info_list = []
    for i in range(world_size):
        with open(args.work_dir + f'/output/instruction_eval_info_list_{i}.json', 'r', encoding='utf-8') as f:
            list_i = json.load(f)
        info_list += list_i

    generate_text = [entry["generate_text"] for entry in info_list]

    instruction_dataset_name = os.path.splitext(os.path.basename(config["data_config"]["instruction_dataset_repo"]))[0]
    raw_cache_path = f'output/{instruction_dataset_name}_test_instruction_dataset.json'
    with open(raw_cache_path, 'r', encoding='utf-8') as f:
        examples_list = json.load(f)

    print("calculate multi-reference BLEU4 / ROUGE-1 F1...")
    instruction_inference_results = []
    bleu4_list = []
    rouge1_scores = []
    rouge = Rouge()

    for gen_text, example in zip(generate_text, examples_list):
        references = example["answers"]
        gen_text = tokenizer.decode(gen_text, skip_special_tokens=True)

        print("references: ", references)
        print("gen_text: ", gen_text)

        gen_ids = tokenizer(gen_text, add_special_tokens=False)["input_ids"]
        reference_ids = [tokenizer(ref, add_special_tokens=False)["input_ids"] for ref in references]
        bleu4 = sentence_bleu(reference_ids, gen_ids, weights=(0.25, 0.25, 0.25, 0.25))

        example["generate"] = gen_text
        example["bleu4"] = bleu4
        example["exact_match"] = 1.0 if any(gen_text == ref for ref in references) else 0.0

        rouge_candidates = []
        best_reference = ""
        best_rouge = 0.0
        for ref in references:
            try:
                score = rouge.get_scores(gen_text, ref)[0]["rouge-1"]["f"]
            except Exception as e:
                print(f"[Error] Rouge计算失败: {str(e)} | 生成文本: '{gen_text}' | 参考文本: '{ref}'")
                score = 0.0
            rouge_candidates.append(score)
            if score > best_rouge:
                best_rouge = score
                best_reference = ref

        example["rouge-f1-all"] = rouge_candidates
        example["best_reference"] = best_reference
        example["rouge-f1"] = best_rouge

        rouge1_scores.append(best_rouge)
        bleu4_list.append(bleu4)
        instruction_inference_results.append(example)

    avg_bleu4 = np.mean(bleu4_list) if bleu4_list else 0.0
    rouge1_f1 = np.mean(rouge1_scores) if rouge1_scores else 0.0

    print(f"avg_bleu4:{avg_bleu4}")
    print(f"rouge1_f1:{rouge1_f1}")

    with open(args.work_dir + '/output/instruction_brief_eval_info.json', 'w', encoding='utf-8') as f:
        json.dump(f"avg_bleu4:{avg_bleu4}, rouge1_f1:{rouge1_f1}", f, ensure_ascii=False)

    with open(args.work_dir + '/output/instruction_inference_results.json', 'w', encoding='utf-8') as f:
        json.dump(instruction_inference_results, f, ensure_ascii=False, indent=4)
