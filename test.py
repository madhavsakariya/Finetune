import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
MODEL_PATH = "llama32_trained"
MAX_NEW_TOKENS = 200
TEMPERATURE = 0.7
TOP_P = 0.9

# ------------------------------------------------------
# GPU CHECK
# ------------------------------------------------------
if torch.cuda.is_available():
    device = "cuda"
    print("GPU:", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("Running on CPU")

# ------------------------------------------------------
# LOAD TOKENIZER & MODEL
# ------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

model.eval()

print("\nModel loaded. Type your prompt below.")
print("Type 'exit' to quit.\n")

# ------------------------------------------------------
# INTERACTIVE LOOP
# ------------------------------------------------------
while True:
    instruction = input(">> Instruction: ").strip()
    if instruction.lower() in ["exit", "quit"]:
        print("Exiting...")
        break

    input_text = input(">> Input (optional, press Enter to skip): ").strip()

    prompt = f"""### Instruction:
{instruction}

### Input:
{input_text}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            eos_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    print("\n===== MODEL RESPONSE =====")
    print(response)
    print("==========================\n")
