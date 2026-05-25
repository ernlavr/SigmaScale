import dotenv
dotenv.load_dotenv("/shared/<REDACTED>/SpinQuant/.env")  # Load environment variables from .env file if present

import torch
import json
from tqdm import tqdm
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import pandas as pd
import random
import numpy as np

def set_random_seeds(seed=42):
    print(f"Setting random seed to {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    return seed
set_random_seeds()

def get_unprocessed_examples(raw_ds, repo_id, branch="revert"):
    """
    Loads the already-processed dataset from HuggingFace and returns
    only the examples from raw_ds that haven't been processed yet,
    matched by 'instruction'.
    
    Args:
        raw_ds: The full source dataset (tatsu-lab/alpaca, already shuffled)
        repo_id: HuggingFace dataset repo, e.g. "/Alpaca-Llama3.1-KD"
        branch: The branch/revision to load from (default: "revert")
    
    Returns:
        Filtered raw_ds with already-processed instructions removed
    """
    print(f"Loading processed dataset from {repo_id} (branch: {branch})...")
    try:
        processed_ds = load_dataset(repo_id, split="train", revision=branch)
        processed_instructions = set(processed_ds["instruction"])
        print(f"Found {len(processed_instructions)} already-processed instructions.")
    except Exception as e:
        print(f"Could not load processed dataset: {e}")
        print("Starting from scratch.")
        return raw_ds

    original_size = len(raw_ds)
    remaining_ds = raw_ds.filter(
        lambda example: example["instruction"] not in processed_instructions,
        desc="Filtering out already-processed examples"
    )
    print(f"Resuming: {len(remaining_ds)}/{original_size} examples remaining to process.")
    return remaining_ds, processed_ds

# ==========================================
# 1. Configuration & Auth
# ==========================================
# login(token="your_hf_token_here") # Uncomment and use if pushing to Hub

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SOURCE_DATASET = "tatsu-lab/alpaca"
TARGET_REPO_ID = "<REDACTED>/Alpaca-Llama3.1-KD" # Change this!
BATCH_SIZE = 1 # Adjust based on your VRAM (8 is safe for 24GB)
NUM_VERSIONS_PER_PROMPT = 3

# ==========================================
# 2. Model & Tokenizer Initialization
# ==========================================
print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# CRITICAL FOR HF BATCHING
tokenizer.pad_token = "<|finetune_right_pad_id|>"
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16, # Use bfloat16 for Llama 3.1 stability
)

# ==========================================
# 3. Dataset Loading & Formatting
# ==========================================
print(f"Loading original dataset: {SOURCE_DATASET}...")
raw_ds = load_dataset(SOURCE_DATASET, split="train").shuffle(seed=42) # Shuffle for randomness
raw_ds, processed_ds = get_unprocessed_examples(raw_ds, TARGET_REPO_ID, branch="revert")

def extract_system_prompt(text_field):
    """
    Extracts the original Alpaca preamble from the 'text' field 
    by splitting at the '### Instruction:' marker.
    """
    # The standard Alpaca marker
    marker = "### Instruction:"
    
    if text_field and marker in text_field:
        # Split and grab everything before the marker, stripping whitespace
        return text_field.split(marker)[0].strip()
    
    # Fallback just in case a row is malformed
    return "Below is an instruction that describes a task. Write a response that appropriately completes the request."

def apply_template(example):
    """Dynamically applies the extracted system prompt and the user instruction."""
    # 1. Extract the dynamic system prompt
    system_prompt = extract_system_prompt(example.get('text', ''))
    
    # 2. Build the user content
    user_content = "\n\n### Instruction:\n" + example['instruction']
    if example.get('input') and str(example['input']).strip():
        user_content += f"\n\n### Input:\n{example['input']}"
        
    user_content += "\n\n### Response:\n" # The model will generate after this
    
    # 3. Format into Llama 3.1 structure
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    instruction = example['instruction']
    additional_input = example.get('input', '').strip()
    original_output = example.get('output', '').strip()
    
    return system_prompt, formatted_prompt, instruction, additional_input, original_output

print("Extracting dynamic system prompts and applying templates...")
system_prompts_list = []
prompts_list = []
instruction_list = []
additional_input_list = []
original_output_list = []

for ex in raw_ds:
    sys_prompt, formatted, instruction, additional_input, original_output = apply_template(ex)
    system_prompts_list.append(sys_prompt)
    prompts_list.append(formatted)
    instruction_list.append(instruction)
    additional_input_list.append(additional_input)
    original_output_list.append(original_output)

    
# ==========================================
# 4. Batched Generation
# ==========================================
terminators = [
    tokenizer.eos_token_id, 
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# dataframe with columns, id, retry_count, instruction, input, output, text
final_df = pd.DataFrame({
    "id": pd.Series(dtype="int"),
    "retry_count": pd.Series(dtype="int"),
    "instruction": pd.Series(dtype="string"),
    "input": pd.Series(dtype="string"),
    "output_llama": pd.Series(dtype="string"),
    "output_original": pd.Series(dtype="string"),
    "text": pd.Series(dtype="string"),
})


print(f"Generating teacher responses in batches of {BATCH_SIZE}...")
for i in tqdm(range(0, len(prompts_list), BATCH_SIZE)):
    batch = prompts_list[i : i + BATCH_SIZE]
    batch_instructions = instruction_list[i : i + BATCH_SIZE]
    batch_inputs = additional_input_list[i : i + BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
    batch_original_outputs = original_output_list[i : i + BATCH_SIZE]
    
    for j in range(NUM_VERSIONS_PER_PROMPT):  # Retry mechanism for robustness    
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
                eos_token_id=terminators,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            
        generated_tokens = outputs[:, inputs.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        batch_rows = []

        for k, (prompt, instr, inp, out, orig_out) in enumerate(
            zip(batch, batch_instructions, batch_inputs, decoded, batch_original_outputs)
        ):
            batch_rows.append({
                "id": i + k,
                "retry_count": j,
                "instruction": instr,
                "input": inp,
                "output_llama": out,
                "output_original": orig_out,
                "text": prompt,
            })

        batch_df = pd.DataFrame(batch_rows)

        final_df = pd.concat([final_df, batch_df], ignore_index=True)
    
    if i % 500 == 0:
        print(f"Processed {i} prompts so far. \n\nInstr: \n{instr}; \n\nInput:\n{inp}\n\nOutput:\n{decoded[0]}\n")
        
    if (i + 1) % 1000 == 0:
        try:
            final_df.to_json("./output_dir/data/alpaca_llama3_1_KD_dataset_with_sys_prompts.json", orient="records", indent=4)
            print("Saved intermediate DataFrame with system prompts to JSON.")
            
            login(token="<REDACTED>")
            new_ds = Dataset.from_pandas(final_df)

            print(f"Uploading dataset to {TARGET_REPO_ID}...")
            new_ds.push_to_hub(TARGET_REPO_ID, split="train")
        except Exception as e:
            print(f"Error during intermediate save/upload: {e}")
            print("Continuing with next batches without interruption...")

    
        

# At the end (step 5), merge with existing before pushing
try:
    existing_ds = load_dataset(TARGET_REPO_ID, split="train", revision="revert")
    existing_df = existing_ds.to_pandas()
    final_df = pd.concat([existing_df, final_df], ignore_index=True).drop_duplicates(subset=["instruction", "retry_count"])
    print(f"Merged with existing data. Total rows: {len(final_df)}")
except Exception as e:
    print(f"Could not load existing data for merge: {e}")

# ==========================================
# 5. Create and Save New Dataset
# ==========================================
# We now save the dynamically extracted system prompt per row
final_df.to_json("../output_dir/data/alpaca_llama3_1_KD_dataset_with_sys_prompts.json", orient="records", indent=4)
print("Saved intermediate DataFrame with system prompts to JSON.")

login(token="<REDACTED>")
new_ds = Dataset.from_pandas(final_df)

print(f"Uploading dataset to {TARGET_REPO_ID}...")
new_ds.push_to_hub(TARGET_REPO_ID, split="train")