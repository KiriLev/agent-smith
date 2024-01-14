import argparse
import json
import logging
import pdb
import pickle

import hydra
import torch
from tqdm import tqdm
from dataloader import MultiChoiceDataset, get_data_split
from hydra.core.hydra_config import HydraConfig
from metric import ActionEvaluatorGeneration, ActionEvaluatorMultiChoice
from omegaconf import DictConfig, OmegaConf
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logger.info(f"Preparing train data for saving to {cfg.output_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)

    # Load train data
    train_data = get_data_split(
        cfg.data.data_path,
        cfg.data.train_split_file,
    )
    train_dataset = MultiChoiceDataset(
        train_data,
        tokenizer,
        neg_ratio=cfg.train.neg_ratio,
        num_candidates=cfg.train.num_candidates,
        max_context_len=cfg.train.max_context_len,
    )

    # Load the llm_prompt format
    with open(cfg.llm_prompt, "r") as f:
        llm_prompt = json.load(f)

    # Initialize variables for file splitting
    file_index = 0
    line_count = 0
    max_lines_per_file = 1000
    jsonl_file = open(f"{cfg.output_path}_part{file_index}.jsonl", 'w')

    for entry in tqdm(train_dataset):
        formatted_entry = {"text": format_entry(entry, llm_prompt, tokenizer)}
        jsonl_file.write(json.dumps(formatted_entry) + "\n")
        line_count += 1

        # Check if the current file has reached its maximum line count
        if line_count >= max_lines_per_file:
            jsonl_file.close()
            file_index += 1
            line_count = 0
            jsonl_file = open(f"{cfg.output_path}/part{file_index}.jsonl", 'w')

    # Close the last file
    jsonl_file.close()
    logger.info(f"Train data saved in multiple files at {cfg.output_path}")

def format_entry(entry, llm_prompt, tokenizer):
    input_text = tokenizer.decode(entry['input_ids'], skip_special_tokens=True)
    labels_text = tokenizer.decode(entry['labels'], skip_special_tokens=True)

    formatted_text = f"<s>[SYSTEM] {llm_prompt[0]['content']}\n[USER] {input_text}\n"

    formatted_text += f"[ASSISTANT] Answer: {labels_text}\n</s>"

    return formatted_text

if __name__ == "__main__":
    main()
