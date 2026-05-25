# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import random
from typing import Any, Dict
from torch.utils.data.distributed import DistributedSampler
import datasets
import torch
import transformers
import os

class TupleDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

def get_wikitext2(nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, mode=None, bs=4):
    print(f"Getting Wikitext-2 dataset with nsamples={nsamples}, seed={seed}, seqlen={seqlen}, model={model}, mode={mode}, bs={bs}")
    
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    if mode == "eval":
        testdata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "test"
        ]
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
        # Remove batch dimension
        input_ids = testenc['input_ids'].squeeze(0)  # Shape: (289007,)
        num_sequences = len(input_ids) // seqlen
        input_ids = input_ids[:num_sequences * seqlen]
        sequences = input_ids.view(num_sequences, seqlen)
        
        output_dir = "/shared/elavrin/SpinQuant/output_dir/data"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(sequences, os.path.join(output_dir, "test_sequences.pt"))
        print(f"Saved test sequences {output_dir}")
        
        dataloader = torch.utils.data.DataLoader(
            sequences,
            batch_size=bs,
            shuffle=False
        )
        
        return dataloader
    elif mode == "train":
        traindata = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "train"
        ]
        trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
        random.seed(seed)
        trainloader = []
        
        if nsamples == "full":
        # Use full dataset sequentially
            total_len = trainenc.input_ids.shape[1]
            for i in range(0, total_len - seqlen, seqlen):
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
                trainloader.append(inp)
        else:
            # Original random sampling
            random.seed(seed)
            for _ in range(nsamples):
                i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
                trainloader.append(inp)

        output_dir = "/shared/elavrin/SpinQuant/output_dir/data"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saved test sequences {output_dir}")

        trainloader = torch.utils.data.DataLoader(
            TupleDataset(trainloader),
            batch_size=bs,
            shuffle=True,  # optional: no shuffle for full pass
        )
        return trainloader
    
    elif mode == "calib":
        seed = seed + nsamples + seqlen  # Ensure a different seed for calibration data
        print(f"Generating calibration data with sampling seed: {seed}")
        calib_data = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")[
            "train"
        ]
        calibenc = tokenizer("\n\n".join(calib_data["text"]), return_tensors="pt")
        random.seed(seed)
        calibloader = []
        for _ in range(nsamples):
            i = random.randint(0, calibenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = calibenc.input_ids[:, i:j]
            calibloader.append(inp)
        
            
        output_dir = "/shared/elavrin/SpinQuant/output_dir/data"
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saved test sequences {output_dir}")    
            
        # create a dataloader
        calibration_data = torch.utils.data.DataLoader(
            TupleDataset(calibloader), 
            batch_size=bs, 
            shuffle=True,
        )
        return calibration_data

def get_c4(nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, eval_mode=False):
    """
    Get a loader for the C4 dataset.
    
    Args:
        nsamples (int): Number of samples to generate.
        seed (int): Random seed for reproducibility.
        seqlen (int): The sequence length of each sample.
        model (str): The model name to load the tokenizer from.
        tokenizer: An already-initialized tokenizer.
        eval_mode (bool): If True, returns the validation set; otherwise, returns a training loader.
        
    Returns:
        If eval_mode is True, a tokenized tensor of the validation data.
        If eval_mode is False, a list of (input, target) tensor pairs for training.
    """
    # Initialize the tokenizer if not provided
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)

    local_dataset_path = "./datasets/c4_en"  # Path where the C4 dataset is stored locally
    # --- Evaluation Mode ---
    if eval_mode:
        # Load a subset of the C4 validation set using streaming
        # valdata = datasets.load_dataset(
        #     'allenai/c4', 'en', split='validation', streaming=True
        # )
        # Load the validation set from the local disk
        valdata = datasets.load_from_disk(os.path.join(local_dataset_path, 'validation'))
        val_text = "\n\n".join(valdata["text"])
        
        # Take the first 10,000 documents for a manageable validation set
        #val_dataset_head = valdata.take(10000)
        #val_text = "\n\n".join([d['text'] for d in val_dataset_head])
        
        valenc = tokenizer(val_text, return_tensors="pt")
        return valenc
        
    # --- Training Mode ---
    else:
        # Load a subset of the C4 training set using streaming
        traindata = datasets.load_dataset(
            'allenai/c4', 'en', split='train', streaming=True
        )

        # Create a large text buffer by taking a fixed number of documents from the stream.
        # This avoids loading the entire massive dataset. 50,000 documents provide
        # a sufficiently large and diverse text corpus to sample from.
        dataset_head = traindata.take(50000) 
        text_samples = [d['text'] for d in dataset_head]
        
        # Concatenate and tokenize the text samples
        train_text = "\n\n".join(text_samples)
        trainenc = tokenizer(train_text, return_tensors="pt")

        # Generate training samples with the same logic as get_wikitext2
        random.seed(seed)
        trainloader = []
        for _ in range(nsamples):
            # Select a random starting point in the tokenized text
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            
            # Extract the input sequence
            inp = trainenc.input_ids[:, i:j]
            
            # Create the target tensor, masking all but the last token
            tar = inp.clone()
            tar[:, :-1] = -100
            
            trainloader.append((inp, tar))
            
        return trainloader

import transformers
import datasets
import torch
import re

def intersect_with_cleaned_alpaca(dataset):
    clean_alpaca = datasets.load_dataset("yahma/alpaca-cleaned", split="train")
    
    def filter_func(original_example, clean_instructions):
        instruction = original_example["instruction"]
        instruction = instruction.replace(" ", "").lower()  # Normalize the instruction for better matching
        instruction = re.sub(r'[^a-zA-Z ]', '', instruction)  # Remove non-alphabetic characters
        
        for clean_instr in clean_instructions:
            if instruction in clean_instr:
                return True
            
            if clean_instr in instruction:
                return True

        return False


    # select datapoints from incoming dataset which "instruction" field matches the "instruction" field in the cleaned alpaca
    clean_instructions = clean_alpaca["instruction"]
    # clean_instructions = [instr.strip().lower().replace(" ", "") for instr in clean_instructions]  # Use a set for O(1) lookups and strip whitespace
    # clean_instructions = [re.sub(r'[^a-zA-Z ]', '', instr) for instr in clean_instructions]
    
    # normalize original
    cleaned_dataset = dataset['instruction']
    # cleaned_dataset = [instr.strip().lower().replace(" ", "") for instr in cleaned_dataset]
    # cleaned_dataset = [re.sub(r'[^a-zA-Z ]', '', instr) for instr in cleaned_dataset]
    
    intersection = set(clean_instructions).intersection(set(cleaned_dataset))
    
    filtered_dataset = dataset.filter(lambda x: filter_func(x, clean_instructions), desc="Filtering with cleaned Alpaca instructions")
    return filtered_dataset
    
    

def get_alpaca(nsamples=128, seed=0, seqlen=2048, model="", tokenizer=None, mode=None, bs=4):
    print(f"Getting Alpaca dataset with nsamples={nsamples}, seed={seed}, seqlen={seqlen}, model={model}, mode={mode}, bs={bs}")
    
    if tokenizer is None:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model, use_fast=False)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    def process_data(example):
        sys_prompt = example['text'].split("\n\n### Instruction")[0]
        instruction = example['instruction']
        
        # Renamed from 'input' to avoid shadowing the built-in Python function
        input_text = example['input'] 
        
        # Note: Chat templates handle role formatting. You might not need the 
        # "### Instruction:" headers if your chat template already formats user/system turns.
        content_string = f"\n\n### Instruction: \n {instruction}"
        if input_text.strip() != "":
            content_string += f"\n\n### Input: \n {input_text}"
        
        # 1. Build messages for the PROMPT ONLY
        prompt_messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": content_string}
        ]
        
        # 2. Build messages for the FULL CONVERSATION
        full_messages = prompt_messages + [
            {"role": "assistant", "content": example['output']}
        ]
            
        # Apply the chat template to both
        # add_generation_prompt=True adds the assistant header (e.g., <|im_start|>assistant\n)
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        full_text = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)

        return {"prompt_text": prompt_text, "full_text": full_text}

    dataset = datasets.load_dataset("tatsu-lab/alpaca", split="train")
    dataset = dataset.shuffle(seed=seed)
    if nsamples != "full":
        dataset = dataset.select(range(nsamples))
    
    # Process the dataset
    dataset = dataset.map(process_data, num_proc=4, desc="Preparing Alpaca dataset")
    
    def tokenize_function(example):
        # Tokenize the full conversation. This is what the model actually sees.
        full_enc = tokenizer(example["full_text"], truncation=True, max_length=seqlen)
        input_ids = full_enc["input_ids"]
        
        # Tokenize JUST the prompt to find out how many tokens it takes up
        prompt_enc = tokenizer(example["prompt_text"], truncation=True, max_length=seqlen)
        prompt_len = len(prompt_enc["input_ids"])
        
        # Mask the prompt with -100, but use the real input_ids for the response
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        
        # Truncate labels just in case the prompt alone exceeded seqlen
        labels = labels[:seqlen]
        
        return {
            "input_ids": input_ids, 
            "attention_mask": full_enc["attention_mask"],
            "labels": labels
        }

    # Tokenize the dataset
    dataset = dataset.map(tokenize_function, num_proc=4, desc="Tokenizing Alpaca dataset")
    dataset = dataset.remove_columns(["instruction", "input", "output", "text", "prompt_text", "full_text"])
    
    # Data collator handles the padding natively
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
        label_pad_token_id=-100 
    )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, collate_fn=collator)
    return dataloader

def get_alpaca_llama31(nsamples=128, seed=0, seqlen=2048, tokenizer=None, bs=4, num_paraphrases_trainset=1):
    print(f"Getting ernlavr/Alpaca-Llama3.1-KD dataset with Llama 3.1 formatting, nsamples={nsamples}, seed={seed}, seqlen={seqlen}, num paraphrases={num_paraphrases_trainset}")
    
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    raw_ds = datasets.load_dataset("ernlavr/Alpaca-Llama3.1-KD", split="train").shuffle(seed=seed)
    
    raw_ds = raw_ds.filter(
        lambda x: x['retry_count'] < num_paraphrases_trainset,
        desc=f"Filtering retry_count < {num_paraphrases_trainset}"
    )
    
    if nsamples != "full":
        raw_ds_selected = raw_ds.select(range(nsamples))
        # add corresponding paraphrases for the selected samples
        selected_ids = set(raw_ds_selected["id"])
        raw_ds = raw_ds.filter(
            lambda x: x['id'] in selected_ids,
            desc=f"Selecting samples with id in {selected_ids}"
        )
    
    
    def process_data(example, tokenizer):
        system_prompt = example['text'].split("\n\n### Instruction")[0]
        system_prompt = tokenizer.decode(tokenizer(system_prompt)['input_ids'], skip_special_tokens=True).split("\n\n")[-1].replace(".user",".")
        instruction = example['instruction']
        input_text = example.get('input', '').strip()
        output_llama = example.get('output_llama', '').strip()
        
        user_content = f"\n\n### Instruction: \n {instruction}"
        if input_text != "":
            user_content += f"\n\n### Input: \n {input_text}"
            
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
        
        full_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_llama}
        ]
        full_prompt = tokenizer.apply_chat_template(full_prompt, tokenize=False, add_generation_prompt=False)
        return {"prompt_text": prompt_text, "full_prompt": full_prompt}
    
    raw_ds = raw_ds.map(lambda x: process_data(x, tokenizer), desc="Extracting system prompts and user content")
    
    def tokenize(example):
        full_enc = tokenizer(example["full_prompt"], truncation=True, max_length=seqlen)
        input_ids = full_enc["input_ids"]
        
        prompt_enc = tokenizer(example["prompt_text"], truncation=True, max_length=seqlen)
        prompt_len = len(prompt_enc["input_ids"])
        
        labels = [-100] * prompt_len + input_ids[prompt_len:]
        labels = labels[:seqlen]
        
        return {
            "input_ids": input_ids, 
            "attention_mask": full_enc["attention_mask"],
            "labels": labels
        }        
    
    tokenized_ds = raw_ds.map(tokenize, desc="Tokenizing Alpaca Llama 3.1 dataset")
    tokenized_ds = tokenized_ds.remove_columns(['id', 'retry_count', 'instruction', 'input', 'output_llama', 'output_original', 'text', 'prompt_text', 'full_prompt'])
    
    collator = transformers.DataCollatorForSeq2Seq(
        tokenizer,
        return_tensors="pt",
        padding=True,
        label_pad_token_id=-100
    )
    dataloader = torch.utils.data.DataLoader(tokenized_ds, batch_size=bs, collate_fn=collator)
    return dataloader
    
    

def get_train_data(args, tokenizer, mode=None):
    if args.train_data == "wikitext2":
        return get_wikitext2(
            nsamples=args.train_loader_nsamples,
            seed=args.seed,
            seqlen=args.train_loader_seqlen,
            tokenizer=tokenizer,
            mode=mode,
            bs=args.train_bs,
        )
    elif args.train_data == "alpaca":
        return get_alpaca(
            nsamples=args.train_loader_nsamples,
            seed=args.seed,
            seqlen=args.train_loader_seqlen,
            tokenizer=tokenizer,
            mode=mode,
            bs=args.train_bs,
        )
    elif args.train_data == "ernlavr/Alpaca-Llama3.1-KD":
        return get_alpaca_llama31(
            nsamples=args.train_loader_nsamples,
            seed=args.seed,
            seqlen=args.train_loader_seqlen,
            tokenizer=tokenizer,
            bs=args.train_bs,
            num_paraphrases_trainset=args.num_paraphrases_trainset
        )
        
def get_test_data(args, tokenizer, mode=None):
    return get_wikitext2(
            nsamples=args.test_loader_nsamples,
            seed=args.seed,
            seqlen=args.test_loader_seqlen,
            tokenizer=tokenizer,
            mode=mode,
            bs=args.eval_bs,
        )
        
def get_calib_data(ptq_args, tokenizer):
    return get_wikitext2(
            seed=ptq_args.seed,
            nsamples=32,
            seqlen=ptq_args.test_loader_seqlen,
            tokenizer=tokenizer,
            mode="calib",
        )


class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size: int = 1024) -> None:
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> Dict[str, Any]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result
