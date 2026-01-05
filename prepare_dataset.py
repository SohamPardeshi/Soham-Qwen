import os

# =========================
# CONFIGURATION (EDIT THIS)
# =========================

INPUT_DIRECTORIES = [
    "./data/cleaned",
]

OUTPUT_FILE = "./data/output/dataset.jsonl"

# =========================
# IMPLEMENTATION
# =========================

def merge_jsonl_files(input_dirs, output_file):
    jsonl_files = []

    for input_dir in input_dirs:
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.lower().endswith(".jsonl"):
                    jsonl_files.append(os.path.join(root, file))

    if not jsonl_files:
        print("No .jsonl files found.")
        return

    jsonl_files.sort()

    total_lines = 0

    with open(output_file, "w", encoding="utf-8") as outfile:
        for path in jsonl_files:
            with open(path, "r", encoding="utf-8") as infile:
                for line in infile:
                    line = line.strip()
                    if line:
                        outfile.write(line + "\n")
                        total_lines += 1

    print(f"Merged {len(jsonl_files)} files into:")
    print(output_file)
    print(f"Total JSONL records written: {total_lines}")

if __name__ == "__main__":
    merge_jsonl_files(INPUT_DIRECTORIES, OUTPUT_FILE)
