import os
import torch
import transformers
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

transformers.logging.set_verbosity_info()

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"
HF_TOKEN = "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"  # read-only token

model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype="auto",
    device_map="auto",
    token=HF_TOKEN,
)

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    token=HF_TOKEN,
)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Explain the difference between supervised and unsupervised learning."}
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=256,
)

generated_ids_trimmed = [
    out_ids[len(in_ids):]
    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
)

print(output_text[0])
