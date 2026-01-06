from finetune import TrainConfig, run_training
from multiprocessing import freeze_support

def main():
    cfg = TrainConfig(
        data_path="data/output/dataset.jsonl",
        out_dir="model/output/qwen3vl_soham_lora",
        # rebuild_tokenized_cache = True, # Only if the dataset.jsonl has changed
        batch_size=512,
        dry_run=False,
        max_debug_examples=5,
        print_raw_examples=False,
        inspect_labels=False,
    )

    run_training(cfg)

if __name__ == "__main__":
    freeze_support()  # required on Windows for multiprocessing spawn safety
    main()