import json
import os
import sys

from model.modeling import get_model, load_adapter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import BASE_PATH

sys.path.append(BASE_PATH)
import matplotlib.pyplot as plt
import torch
from rouge import Rouge
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import numpy as np
import argparse
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import logging
from nltk.translate.bleu_score import sentence_bleu

from torch.nn import DataParallel
import torch.multiprocessing as mp



# Launch multi-process eval
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    work_dir = "/mnt/zhaorunsong/lx/SAC-TTT/experiment/full_train_full_test/pwc_vanilla"
    with open(work_dir + "/output/config.json") as f:
        config = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(config["data_config"]["model_id"])
    training_config = config["sft_training_config"]
    task_config = config["sft_task_config"]
    model = get_model(training_config["model_id"], task_config, 0)
    model = load_adapter(model, save_path_and_name=work_dir + '/output/instruction_adapter.pt', log=True)
    model.eval()


    context = """"French senior civil servant arrested on suspicion of spying for North Korea\n\nNovember 27, 2018 by Joseph Fitsanakis\n\nA senior civil servant in the upper house of the French parliament has been arrested on suspicion of spying for North Korea, according to prosecutors. The news of the suspected spy’s arrest was first reported on Monday by Quotidien, a daily politics and culture show on the Monaco-based television channel TMC. The show cited “a judicial source in Paris” and said that France’s domestic security and counterintelligence agency, the General Directorate for Internal Security (DGSI), was in charge of the espionage case.\n\nThe senior administrator has been identified as Benoit Quennedey, a civil servant who liaises between the French Senate and the Department of Architecture and Heritage, which operates under France’s Ministry of Culture. Quennedey was reportedly detained on Sunday morning and his office in the French Senate was raided by DGSI officers on the same day. Quotidien said that he was arrested on suspicion of “collecting and delivering to a foreign power information likely to subvert core national interests”. The report did not provide specific information about the type of information that Quennedey is believed to have passed to North Korea. It did state, however, that a counterintelligence investigation into his activities began in March of this year.\n\nQuennedey is believed to be the president of the Franco-Korean Friendship Association, the French branch of a Spanish-based organization that lobbies in favor of international support for North Korea. Korea Friendship Association branches exist in over 30 countries and are believed to be officially sanctioned by Pyongyang. They operate as something akin to the pre-World War II Comintern (Communist International), a Moscow-sanctioned international pressure group that advocated in favor of Soviet-style communism around the world. French media reported on Monday that Quennedey traveled extensively to the Korean Peninsula in the past decade and has written a French-language book on North Korea. News reports said that the French President Emmanuel Macron had been made aware of Quennedey’s arrest. The senior civil servant faces up to 30 years in prison if found guilty of espionage.\n\n► Author: Joseph Fitsanakis | Date: 27 November 2018 | Permalink\n\n"""
    prompt = "Identify the person arrested on suspicion of spying for North Korea."
    context = tokenizer(context, add_special_tokens=False)["input_ids"]
    prompt = tokenizer(prompt, add_special_tokens=False)["input_ids"]

    context_ids = [tokenizer.bos_token_id] + tokenizer("### Context:\n", add_special_tokens=False)[
        "input_ids"] + context
    question_ids = tokenizer("\n### Question:\n", add_special_tokens=False)["input_ids"] + prompt \
                   + tokenizer("\n### Answer:\n", add_special_tokens=False)["input_ids"]
    inputs = torch.LongTensor(context_ids).unsqueeze(0).cuda()
    lm_target = torch.LongTensor(question_ids).unsqueeze(0).cuda()
    examples={"input_ids": inputs, "lm_targets": lm_target}
    generate_text = model.lm_inference(examples)
    print("generate_text: ", tokenizer.decode(generate_text, skip_special_tokens=True))

