import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling

HF_TOKEN = "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"

@dataclass
class TrainConfig:
    # Model / auth
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    hf_token_env: str = None

    # IO
    data_path: str = "data/output/dataset.jsonl"
    out_dir: str = "model/output/qwen3vl_soham_lora"

    # Dataset
    seed: int = 42
    eval_split_ratio: float = 0.02

    # Training
    max_length: int = 2048
    packing: bool = False
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    save_total_limit: int = 2
    enable_gradient_checkpointing: bool = True

    # Precision
    bf16: bool = True
    fp16: bool = False

    # QLoRA / bitsandbytes
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

    # LoRA
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Debug / inspection modes
    dry_run: bool = False
    max_debug_examples: int = 3
    print_raw_examples: bool = False
    print_formatted_examples: bool = False
    inspect_collator: bool = False


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise EnvironmentError("GPU not found. Qwen3-VL-8B requires a GPU to finetune.")

    print("Environment")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  bitsandbytes: {bnb.__version__}")
    print("")


def _get_hf_token(cfg: TrainConfig) -> Optional[str]:
    return HF_TOKEN


def _load_processor_and_tokenizer(model_id: str, token: Optional[str]):
    print("Loading processor/tokenizer")
    processor = AutoProcessor.from_pretrained(model_id, token=token)
    tokenizer = processor.tokenizer

    # Ensure EOS token is set; Qwen chat templates often rely on <|im_end|>
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    print(f"  EOS token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
    print("")
    return processor, tokenizer


def _build_quant_config(cfg: TrainConfig) -> BitsAndBytesConfig:
    compute_dtype = torch.bfloat16 if cfg.bf16 and torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
    )


def _load_base_model(cfg: TrainConfig, token: Optional[str]):
    print("Loading base model (4-bit QLoRA)")
    quant_cfg = _build_quant_config(cfg)

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg.model_id,
        device_map="auto",
        quantization_config=quant_cfg,
        torch_dtype="auto",
        token=token,
    )

    if cfg.enable_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing enabled")

    print("")
    return model


def _attach_lora(cfg: TrainConfig, model):
    print("Adding LoRA adapters")
    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(cfg.lora_target_modules),
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    print("")
    return model


def _load_and_split_dataset(cfg: TrainConfig):
    print("Loading dataset")
    ds = load_dataset("json", data_files={"train": cfg.data_path}, split="train")

    split = ds.train_test_split(test_size=cfg.eval_split_ratio, seed=cfg.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    print(f"  Train size: {len(train_ds)}")
    print(f"  Eval size:  {len(eval_ds)}")
    print("")
    return train_ds, eval_ds


def to_qwen3vl_message_segments(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts [{"role": "...", "content": "..."}] into Qwen3-VL multimodal segments:
      [{"role": "...", "content": [{"type":"text","text":"..."}]}]
    """
    out = []
    for m in messages:
        out.append(
            {
                "role": m["role"],
                "content": [{"type": "text", "text": m.get("content", "")}],
            }
        )
    return out


def build_formatting_func(processor):
    def formatting_func(example: Dict[str, Any]) -> str:
        msgs = to_qwen3vl_message_segments(example["messages"])
        return processor.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=False,
        )
    return formatting_func


def _build_collator(tokenizer):
    pad_token = tokenizer.pad_token or tokenizer.eos_token
    pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)

    return DataCollatorForLanguageModeling(
        pad_token_id=pad_token_id,
        completion_only_loss=True,
        padding_free=False,
    )


def _build_trainer(cfg: TrainConfig, model, tokenizer, processor, train_ds, eval_ds):
    print("Building trainer")
    args = SFTConfig(
        output_dir=cfg.out_dir,
        max_length=cfg.max_length,
        packing=cfg.packing,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        eval_strategy="steps",
        save_total_limit=cfg.save_total_limit,
        bf16=cfg.bf16 and torch.cuda.is_available(),
        fp16=cfg.fp16,
        report_to=[],
    )

    formatting_func = build_formatting_func(processor)
    collator = _build_collator(tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer, # ALTERNATIVELY: try processor
        formatting_func=formatting_func,
        data_collator=collator,
    )

    print("")
    return trainer, formatting_func, collator


def _debug_print_examples(cfg: TrainConfig, train_ds, formatting_func) -> None:
    n = min(cfg.max_debug_examples, len(train_ds))

    if cfg.print_raw_examples:
        print("Raw training examples")
        for i in range(n):
            ex = train_ds[i]
            print(f"\n--- example {i} ---")
            print(ex)

    if cfg.print_formatted_examples:
        print("\nFormatted training examples (chat template output)")
        for i in range(n):
            ex = train_ds[i]
            text = formatting_func(ex)
            print(f"\n--- formatted {i} ---")
            print(text)


def _debug_inspect_collator(
    cfg: TrainConfig,
    train_ds,
    formatting_func,
    collator,
    tokenizer,
) -> None:
    if not cfg.inspect_collator:
        return

    n = min(cfg.max_debug_examples, len(train_ds))
    print("\nCollator inspection (input_ids / labels masking)")

    # 1) Format to chat-templated text
    texts = [formatting_func(train_ds[i]) for i in range(n)]

    # 2) Tokenize to produce input_ids / attention_mask
    # Use padding=False here; the collator will pad.
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=cfg.max_length,
        padding=False,
        return_attention_mask=True,
        add_special_tokens=False,  # important: chat template already includes special tokens
    )

    # 3) Build features list the way TRL collator expects
    examples = []
    for i in range(n):
        examples.append(
            {
                "input_ids": tokenized["input_ids"][i],
                "attention_mask": tokenized["attention_mask"][i],
            }
        )

    # 4) Collate (this will create labels and apply completion-only masking)
    out = collator(examples)

    input_ids = out["input_ids"]
    labels = out["labels"]

    for i in range(n):
        ids = input_ids[i].tolist()
        labs = labels[i].tolist()

        masked = sum(1 for x in labs if x == -100)
        unmasked = len(labs) - masked

        print(f"\n--- batch item {i} ---")
        print(f"  seq_len:  {len(ids)}")
        print(f"  masked:   {masked}")
        print(f"  unmasked: {unmasked}")

        preview_n = min(180, len(ids))
        decoded_preview = tokenizer.decode(ids[:preview_n], skip_special_tokens=False)
        print("\n  decoded preview (start of sequence):")
        print(decoded_preview)

        first_unmasked = next((j for j, x in enumerate(labs) if x != -100), None)
        print(f"\n  first unmasked label index: {first_unmasked}")

        if first_unmasked is not None:
            # show a small window around where loss starts
            start = max(0, first_unmasked - 40)
            end = min(len(ids), first_unmasked + 140)
            window_ids = ids[start:end]
            window_labs = labs[start:end]

            window_text = tokenizer.decode(window_ids, skip_special_tokens=False)
            print("\n  decoded window around first unmasked token:")
            print(window_text)

            # show how many tokens in this window are masked/unmasked
            w_masked = sum(1 for x in window_labs if x == -100)
            w_unmasked = len(window_labs) - w_masked
            print(f"\n  window masked: {w_masked}, window unmasked: {w_unmasked}")


def run_training(cfg: TrainConfig) -> None:
    """
    End-to-end finetuning entrypoint (callable from elsewhere).
    Set cfg.dry_run=True to verify everything and print debug output without training.
    """
    _require_cuda()
    token = _get_hf_token(cfg)

    processor, tokenizer = _load_processor_and_tokenizer(cfg.model_id, token)
    model = _load_base_model(cfg, token)
    model = _attach_lora(cfg, model)

    train_ds, eval_ds = _load_and_split_dataset(cfg)

    trainer, formatting_func, collator = _build_trainer(
        cfg=cfg,
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        train_ds=train_ds,
        eval_ds=eval_ds,
    )

    _debug_print_examples(cfg, train_ds, formatting_func)
    _debug_inspect_collator(cfg, train_ds, formatting_func, collator, tokenizer)

    if cfg.dry_run:
        print("\nDry run enabled. Exiting before training.")
        return

    print("Starting training")
    trainer.train()

    print("Saving adapter and processor")
    trainer.save_model(cfg.out_dir)
    processor.save_pretrained(cfg.out_dir)

    print(f"Done. Saved to: {cfg.out_dir}")
