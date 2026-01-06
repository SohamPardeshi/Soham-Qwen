from __future__ import annotations

import re
import shutil
from pathlib import Path

# Hardcode as requested (raw string to avoid backslash escaping issues on Windows).
SEARCH_PATH = r"C:\Users\soham\Downloads\fb"
OUTPUT_DIR = r".\data\raw"

# Relative location inside the Facebook export.
E2EE_REL = Path("your_facebook_activity") / "messages" / "e2ee_cutover"

# Match message_<digits>.json exactly, e.g. message_3.json, message_12.json
MESSAGE_NUM_RE = re.compile(r"^message_(\d+)\.json$", re.IGNORECASE)


def unique_destination_path(dest: Path) -> Path:
    """
    If dest exists, produce a non-colliding path by appending _dupN before the suffix.
    """
    if not dest.exists():
        return dest

    stem = dest.stem
    suffix = dest.suffix
    i = 2
    while True:
        candidate = dest.with_name(f"{stem}_dup{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def main() -> None:
    base = Path(SEARCH_PATH)
    e2ee_root = base / E2EE_REL

    if not e2ee_root.exists() or not e2ee_root.is_dir():
        raise FileNotFoundError(f"Expected directory not found: {e2ee_root}")

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    conversations = 0

    # Enumerate immediate subfolders of e2ee_cutover
    for convo_dir in sorted([p for p in e2ee_root.iterdir() if p.is_dir()]):
        conversations += 1
        convo_name = convo_dir.name

        # Look for files in that folder; you said "one or more message.json files"
        # and you specifically want message_#.json, so we filter for that.
        for f in convo_dir.iterdir():
            if not f.is_file():
                continue

            m = MESSAGE_NUM_RE.match(f.name)
            if not m:
                # Skip message.json and anything else not matching message_<digits>.json
                skipped += 1
                continue

            msg_num = m.group(1)  # digits as string
            new_name = f"message_{convo_name}_{msg_num}.json"
            dest = out_dir / new_name
            dest = unique_destination_path(dest)

            shutil.copy2(f, dest)
            copied += 1

    print("Done.")
    print(f"Source root: {e2ee_root}")
    print(f"Output dir:  {out_dir}")
    print(f"Conversations scanned: {conversations}")
    print(f"Files copied: {copied}")
    print(f"Entries skipped (non-matching files): {skipped}")


if __name__ == "__main__":
    main()
