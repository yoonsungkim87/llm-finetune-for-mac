import os
import json
import torch
from random import randint
from datasets import load_dataset
from huggingface_hub import hf_hub_download, login
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTConfig, SFTTrainer
import shutil
import glob
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
MODEL_ID = "google/functiongemma-270m-it"
DATASET_ID = "google/mobile-actions"
OUTPUT_DIR = "result"
LITERTLM_OUTPUT_DIR = "litertlm_output"
HF_TOKEN = os.environ.get("HF_TOKEN")

# --- 1. Setup & Device ---
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" # Enable full memory usage on MPS

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

device = get_device()
print(f"Using device: {device}")

if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    print("Warning: HF_TOKEN environment variable not set. Please ensure you are logged in using `huggingface-cli login`.")

# --- 2. Load & Prepare Dataset ---
print("Loading and processing dataset...")
data_file = hf_hub_download(repo_id=DATASET_ID, filename="dataset.jsonl", repo_type="dataset")
dataset = load_dataset("text", data_files=data_file, encoding="utf-8")["train"].shuffle(seed=42)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

def apply_format(sample):
    template_inputs = json.loads(sample['text'])
    
    # Prompt (excluding the last assistant message)
    prompt = tokenizer.apply_chat_template(
        template_inputs['messages'][:-1],
        tools=template_inputs['tools'],
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Full conversation
    prompt_and_completion = tokenizer.apply_chat_template(
        template_inputs['messages'],
        tools=template_inputs['tools'],
        tokenize=False,
        add_generation_prompt=False
    )
    
    completion = prompt_and_completion[len(prompt):]
    
    return {
        "prompt": prompt,
        "completion": completion,
        "split": template_inputs["metadata"],
    }

processed_dataset = dataset.map(apply_format)

# Calculate max length
longest_example = max(processed_dataset, key=lambda x: len(x['prompt'] + x['completion']))
max_token_count = len(tokenizer.tokenize(longest_example['prompt'] + longest_example['completion'])) + 100
print(f"Max token count set to: {max_token_count}")

train_dataset = processed_dataset.filter(lambda x: x['split'] == 'train')
eval_dataset = processed_dataset.filter(lambda x: x['split'] == 'eval')

# --- 3. Fine-tuning ---
print("Starting fine-tuning...")

# Training Args adapted for Mac
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=2,
    per_device_train_batch_size=1,  # Reduced from 4 to 1
    gradient_accumulation_steps=32, # Increased from 8 to 32
    logging_steps=50,
    save_strategy="epoch",
    learning_rate=1e-5,
    max_length=512, # Reduced for MPS stability
    packing=False,
    # MPS support for bf16 is limited/mixed, fp16 is safer usually, or float32.
    # trying bf16=True if on Apple Silicon, otherwise False.
    bf16=False, 
    fp16=False, # Disable fp16 on Mac to avoid unscale error
    optim="adamw_torch", # Fused optimizer might be CUDA specific
    report_to="none",
    dataset_text_field="text", # Dummy field, we use formatted dataset
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map=device,
    torch_dtype=torch.float32, # Use float32 for MPS stability
)
base_model.config.pad_token_id = tokenizer.pad_token_id

trainer = SFTTrainer(
    model=base_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Check for existing checkpoints
checkpoints = glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*"))
if checkpoints:
    print(f"Found {len(checkpoints)} checkpoints. Resuming from the latest one.")
    trainer.train(resume_from_checkpoint=True)
else:
    print("No checkpoints found. Starting training from scratch.")
    trainer.train()

print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
