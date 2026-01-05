import os
import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import bitsandbytes as bnb
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen3VLForConditionalGeneration,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
from torch.nn.utils.rnn import pad_sequence


# -------------------------
# Configuration
# -------------------------

HF_TOKEN = "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"

@dataclass
class TrainConfig:
    # Model / auth
    model_id: str = "Qwen/Qwen3-VL-8B-Instruct"
    hf_token_env: str = None

    # IO
    data_path: str = "data/output/dataset.jsonl"
    out_dir: str = "model/output/qwen3vl_soham_lora"

    # Arrow Caching
    tokenized_cache_dir: str = "data/cache/tokenized_qwen3vl"
    rebuild_tokenized_cache: bool = False

    # Dataset
    seed: int = 42
    eval_split_ratio: float = 0.02

    # Training
    batch_size: int = 512
    max_length: int = 2048
    packing: bool = False
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    logging_steps: int = 50
    save_steps: int = 10000
    eval_steps: int = 10000
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
    inspect_labels: bool = False  # prints assistant-only decoded label span


# -------------------------
# Environment / loading
# -------------------------

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

    # Ensure EOS token is set
    if tokenizer.eos_token is None:
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    print(f"  EOS token: {tokenizer.eos_token} ({tokenizer.eos_token_id})")
    print("")
    return processor, tokenizer


def _build_quant_config(cfg: TrainConfig) -> BitsAndBytesConfig:
    compute_dtype = torch.bfloat16 if (cfg.bf16 and torch.cuda.is_bf16_supported()) else torch.float16
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


# -------------------------
# Dataset tokenization + labels (assistant-only)
# -------------------------

QWEN_IM_START = "<|im_start|>"
QWEN_IM_END = "<|im_end|>\n"  # if your template has no trailing newline, change to "<|im_end|>"

def _encode_piece(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _tokenize_conversation_assistant_only(
    example: Dict[str, Any],
    tokenizer,
    max_length: int,
) -> Dict[str, Any]:
    """
    Builds input_ids + labels directly from message roles.
    Labels are -100 everywhere except assistant content tokens.
    """
    input_ids: List[int] = []
    labels: List[int] = []

    messages: List[Dict[str, Any]] = example["messages"]

    for m in messages:
        role = m["role"]
        content = m.get("content", "")

        # Wrapper tokens, matching Qwen template style
        prefix_text = f"{QWEN_IM_START}{role}\n"
        suffix_text = QWEN_IM_END

        prefix_ids = _encode_piece(tokenizer, prefix_text)

        if role == "assistant":
            content = content.rstrip() + "\n"
        content_ids = _encode_piece(tokenizer, content)
        
        suffix_ids = _encode_piece(tokenizer, suffix_text)

        # Append to input_ids
        input_ids.extend(prefix_ids)
        input_ids.extend(content_ids)
        input_ids.extend(suffix_ids)

        # Append to labels
        if role == "assistant":
            # Mask wrapper tokens, unmask only the content tokens
            labels.extend([-100] * len(prefix_ids))
            labels.extend(content_ids)
            labels.extend([-100] * len(suffix_ids))
        else:
            # Mask everything for system/user turns
            labels.extend([-100] * (len(prefix_ids) + len(content_ids) + len(suffix_ids)))

        # Truncate if needed (keep input_ids and labels aligned)
        if len(input_ids) >= max_length:
            input_ids = input_ids[:max_length]
            labels = labels[:max_length]
            break

    attention_mask = [1] * len(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def _tokenize_conversation_assistant_only_batched(batch: Dict[str, Any], tokenizer, max_length: int) -> Dict[str, Any]:
    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for messages in batch["messages"]:
        ex = {"messages": messages}
        out = _tokenize_conversation_assistant_only(ex, tokenizer, max_length)
        input_ids_list.append(out["input_ids"])
        attention_mask_list.append(out["attention_mask"])
        labels_list.append(out["labels"])

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }


def _load_and_prepare_dataset(cfg: TrainConfig, tokenizer):
    if (not cfg.rebuild_tokenized_cache) and os.path.isdir(cfg.tokenized_cache_dir):
        print(f"Loading tokenized dataset from disk: {cfg.tokenized_cache_dir}")
        ds = load_from_disk(cfg.tokenized_cache_dir)
    else:
        print("Loading raw JSONL dataset")
        ds = load_dataset("json", data_files={"train": cfg.data_path}, split="train")

        is_windows = (os.name == "nt")
        num_proc = 1 if is_windows else max(1, os.cpu_count() - 1)

        ds = ds.map(
            lambda batch: _tokenize_conversation_assistant_only_batched(batch, tokenizer, cfg.max_length),
            batched=True,
            batch_size=cfg.batch_size,
            num_proc=num_proc,
            writer_batch_size=cfg.batch_size,
            remove_columns=ds.column_names,
            desc="Tokenizing + building assistant-only labels (batched)",
        )

        print(f"Saving tokenized dataset to disk: {cfg.tokenized_cache_dir}")
        ds.save_to_disk(cfg.tokenized_cache_dir)

    # OPTIMIZATION: for large dataset, we don't want to shuffle the entire thing in memory
    eval_every = max(1, int(1 / cfg.eval_split_ratio))
    eval_ds = ds.filter(lambda ex, idx: (idx % eval_every) == 0, with_indices=True)
    train_ds = ds.filter(lambda ex, idx: (idx % eval_every) != 0, with_indices=True)

    print(f"  Train size: {len(train_ds)}")
    print(f"  Eval size:  {len(eval_ds)}")
    print("")
    return train_ds, eval_ds


# -------------------------
# Collator (pads input_ids/attention_mask/labels)
# -------------------------

class PadToMaxInBatch:
    def __init__(self, tokenizer):
        pad_token = tokenizer.pad_token or tokenizer.eos_token
        self.pad_token_id = tokenizer.convert_tokens_to_ids(pad_token)

    def __call__(self, features):
        ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
        labs = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        ams = [torch.tensor(f.get("attention_mask", [1] * len(f["input_ids"])), dtype=torch.long) for f in features]

        input_ids = pad_sequence(ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labs, batch_first=True, padding_value=-100)
        attention_mask = pad_sequence(ams, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# -------------------------
# Trainer
# -------------------------

def _build_trainer(cfg: TrainConfig, model, tokenizer, train_ds, eval_ds):
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

    collator = PadToMaxInBatch(tokenizer)

    # Newer TRL versions support skipping dataset preparation when you pass pre-tokenized datasets.
    init_sig = inspect.signature(SFTTrainer.__init__)
    extra_kwargs = {}
    if "dataset_kwargs" in init_sig.parameters:
        extra_kwargs["dataset_kwargs"] = {"skip_prepare_dataset": True}

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=collator,
        **extra_kwargs,
    )

    print("")
    return trainer


# -------------------------
# Debug
# -------------------------

def decode_unmasked_spans(tokenizer, input_ids, labels):
    spans = []
    start = None
    for i, lab in enumerate(labels):
        if lab != -100 and start is None:
            start = i
        if lab == -100 and start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(labels)))

    texts = []
    for (a, b) in spans:
        texts.append(tokenizer.decode(input_ids[a:b], skip_special_tokens=False))
    return texts

def _debug_print_examples(cfg: TrainConfig, train_ds, tokenizer) -> None:
    n = min(cfg.max_debug_examples, len(train_ds))

    if cfg.print_raw_examples:
        print("Raw training examples (tokenized dict keys)")
        for i in range(n):
            ex = train_ds[i]
            print(f"\n--- example {i} ---")
            print({k: (len(v) if isinstance(v, list) else v) for k, v in ex.items()})

    if cfg.inspect_labels:
        print("\nAssistant-only label inspection")
        for i in range(n):
            ex = train_ds[i]
            ids = ex["input_ids"]
            labs = ex["labels"]

            first_unmasked = next((j for j, x in enumerate(labs) if x != -100), None)
            masked = sum(1 for x in labs if x == -100)
            unmasked = len(labs) - masked

            print(f"\n--- example {i} ---")
            print(f"  seq_len: {len(ids)}")
            print(f"  masked: {masked}")
            print(f"  unmasked: {unmasked}")
            print(f"  first unmasked label index: {first_unmasked}")

            # Decode only assistant-supervised tokens
            # unmasked_ids = [tid for tid, lab in zip(ids, labs) if lab != -100]
            # print("\n  decoded UNMASKED span (should be assistant-only content):")
            # print(tokenizer.decode(unmasked_ids, skip_special_tokens=False))

            span_texts = decode_unmasked_spans(tokenizer, ex["input_ids"], ex["labels"])
            print("\n  decoded UNMASKED spans (per assistant chunk):")
            for j, t in enumerate(span_texts):
                print(f"    [span {j}] {repr(t)}")

            # Optional: show first ~200 tokens of the full sequence for context
            preview_n = min(200, len(ids))
            print("\n  decoded preview (start of full sequence):")
            print(tokenizer.decode(ids[:preview_n], skip_special_tokens=False))

        decoded_full = tokenizer.decode(ex["input_ids"], skip_special_tokens=False)
        decoded_unmasked = tokenizer.decode([tid for tid, lab in zip(ex["input_ids"], ex["labels"]) if lab != -100], skip_special_tokens=False)



# -------------------------
# Entry point
# -------------------------

def run_training(cfg: TrainConfig) -> None:
    _require_cuda()
    token = _get_hf_token(cfg)

    processor, tokenizer = _load_processor_and_tokenizer(cfg.model_id, token)
    model = _load_base_model(cfg, token)
    model = _attach_lora(cfg, model)

    train_ds, eval_ds = _load_and_prepare_dataset(cfg, tokenizer)

    _debug_print_examples(cfg, train_ds, tokenizer)

    if cfg.dry_run:
        print("\nDry run enabled. Exiting before training.")
        return

    trainer = _build_trainer(cfg, model, tokenizer, train_ds, eval_ds)

    print("Training prechecks:")
    print("Trainer collator:", type(trainer.data_collator))

    print("Starting training")
    trainer.train()

    print("Saving adapter and processor")
    trainer.save_model(cfg.out_dir)
    processor.save_pretrained(cfg.out_dir)
    print(f"Done. Saved to: {cfg.out_dir}")
