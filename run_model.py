import torch
from transformers import AutoProcessor
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration

# -------------------------
# Global constants
# -------------------------

BASE_MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
ADAPTER_PATH = "model/output/qwen3vl_soham_lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Optional speedups (safe on 3090)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -------------------------
# Load tokenizer / processor
# -------------------------

processor = AutoProcessor.from_pretrained(ADAPTER_PATH)
tokenizer = processor.tokenizer

# -------------------------
# Load base model
# -------------------------

model = Qwen3VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

# -------------------------
# Load LoRA adapter
# -------------------------

model = PeftModel.from_pretrained(model, ADAPTER_PATH)
model.eval()

print("Model + LoRA adapter loaded")

# -------------------------
# Prompt helper
# -------------------------

SYSTEM_PROMPT = (
    "You are Soham Pardeshi, a kind, helpful person. "
    "You are talking to your friend."
)

def build_prompt(user_text: str) -> str:
    """
    Builds a single-turn chat prompt using the same
    Qwen-style template you trained on.
    """
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

@torch.no_grad()
def generate_reply(
    user_text: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.95,
):
    prompt = build_prompt(user_text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,              # IMPORTANT
        temperature=temperature,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    # Slice off the prompt
    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]

    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()

    return text

# -------------------------
# Interactive loop
# -------------------------

if __name__ == "__main__":
    print("\nType a message. Ctrl+C to exit.\n")

    while True:
        user = input("You: ").strip()
        if not user:
            continue

        reply = generate_reply(user)
        print(f"\nSoham: {reply}\n")
