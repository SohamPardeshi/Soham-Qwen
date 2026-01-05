from finetune import TrainConfig, run_training

cfg = TrainConfig(
    data_path="data/output/dataset.jsonl",
    out_dir="model/output/qwen3vl_soham_lora",
    dry_run=True,
    max_debug_examples=5,
    print_raw_examples=False,
    print_formatted_examples=False,
    inspect_collator=True,
)

run_training(cfg)
