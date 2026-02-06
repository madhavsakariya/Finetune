import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
MODEL_NAME = "meta-llama/Llama-3.2-1B"
HF_TOKEN = "**********************************"
DATASET_NAME = "yahma/alpaca-cleaned"
MAX_LENGTH = 512
EPOCHS = 1
BATCH_SIZE = 2

# ------------------------------------------------------
# GPU CHECK
# ------------------------------------------------------
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("NO GPU AVAILABLE")

# ------------------------------------------------------
# MODEL + TOKENIZER
# ------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    torch_dtype=torch.bfloat16,  # IMPORTANT: bf16
    device_map="auto"
)

# ------------------------------------------------------
# LOAD DATASET
# ------------------------------------------------------
dataset = load_dataset(DATASET_NAME)

# ------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------
def tokenize(example):
    text = (
        f"### Instruction:\n{example['instruction']}\n\n"
        f"### Input:\n{example['input']}\n\n"
        f"### Response:\n{example['output']}"
    )
    out = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding="max_length"
    )
    out["labels"] = out["input_ids"].copy()
    return out

tokenized = dataset["train"].map(tokenize)

# ------------------------------------------------------
# COLLATOR
# ------------------------------------------------------
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ------------------------------------------------------
# TRAINING ARGS
# ------------------------------------------------------
training_args = TrainingArguments(
    output_dir="llama32_trained",

    per_device_train_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,

    bf16=True,          # <── use BF16
    fp16=False,         # <── disable FP16 completely
    optim="adamw_torch",

    # Prevent FP16 scaling errors
    gradient_checkpointing=True,
    gradient_accumulation_steps=2,
    fp16_full_eval=False,

    logging_steps=20,
    save_strategy="epoch",
    report_to="none",
)

# ------------------------------------------------------
# TRAINER
# ------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=collator
)

# ------------------------------------------------------
# TRAIN
# ------------------------------------------------------
trainer.train()

# ------------------------------------------------------
# SAVE
# ------------------------------------------------------
model.save_pretrained("llama32_trained")
tokenizer.save_pretrained("llama32_trained")

print("DONE. Model saved!")
