import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

URL_RE = re.compile(r'https?://\S+')
MESSAGE_FILE_RE = re.compile(r"^message_(\d+)\.json$", re.IGNORECASE)

def has_any_key(d: Dict[str, Any], keys: List[str]) -> bool:
    return any(k in d for k in keys)

def looks_like_call_or_system_text(text: str) -> bool:
    t = text.lower().strip()
    # Extend as you find more patterns in your exports
    if t.startswith("you called "):
        return True
    if t.endswith(" called you."):
        return True
    if "missed your" in t and "call" in t:
        return True
    if "video call ended" in t:
        return True
    return False

def normalize_message(
    msg: Dict[str, Any],
    skip_if_has_keys: List[str],
    redact_urls: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    Returns a normalized message event:
      { sender_name, ts_ms, text }
    or None if message should be skipped.
    """
    if has_any_key(msg, skip_if_has_keys):
        return None

    text = msg.get("content")
    if text is None:
        return None

    if looks_like_call_or_system_text(text):
        return None

    if redact_urls:
        text = URL_RE.sub("[LINK]", text)

    # Basic whitespace normalization
    text = " ".join(text.split()).strip()
    if not text:
        return None

    return {
        "sender": msg.get("sender_name", ""),
        "ts_ms": int(msg.get("timestamp_ms", 0)),
        "text": text,
    }

def merge_consecutive_by_sender(
    events: List[Dict[str, Any]],
    merge_window_seconds: int = 3600,
    join_with: str = "\n",
) -> List[Dict[str, Any]]:
    """
    Merge consecutive events from the same sender if time gap <= merge_window_seconds.
    """
    if not events:
        return []

    merged: List[Dict[str, Any]] = []
    window_ms = merge_window_seconds * 1000

    cur = dict(events[0])
    for nxt in events[1:]:
        same_sender = (nxt["sender"] == cur["sender"])
        gap = nxt["ts_ms"] - cur["ts_ms"]  # events are sorted; gap >= 0

        if same_sender and gap <= window_ms:
            cur["text"] = cur["text"] + join_with + nxt["text"]
            cur["ts_ms"] = nxt["ts_ms"]
        else:
            merged.append(cur)
            cur = dict(nxt)

    merged.append(cur)
    return merged

def split_into_sessions(
    events: List[Dict[str, Any]],
    session_gap_seconds: int = 6 * 3600,
) -> List[List[Dict[str, Any]]]:
    """
    Break the timeline into sessions when adjacent gap exceeds session_gap_seconds.
    This is independent from the merge window.
    """
    if not events:
        return []

    sessions: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = [events[0]]
    gap_ms = session_gap_seconds * 1000

    for prev, nxt in zip(events, events[1:]):
        if (nxt["ts_ms"] - prev["ts_ms"]) > gap_ms:
            sessions.append(cur)
            cur = [nxt]
        else:
            cur.append(nxt)

    sessions.append(cur)
    return sessions

def role_for_sender(sender: str, my_name: str) -> str:
    return "assistant" if sender == my_name else "user"

def build_chat_examples_from_session(
    session: List[Dict[str, Any]],
    my_name: str,
    max_messages_per_example: int = 24,
    include_system: bool = True,
    system_text: str = "You are Soham. Speak exactly like Soham.",
) -> List[Dict[str, Any]]:
    """
    Convert a session into one or more training examples.

    Strategy:
    - Create rolling windows of up to max_messages_per_example messages.
    - Each example starts at a boundary that keeps roles alternating as in chat.
    - This yields multiple training examples per session and helps training.
    """
    if not session:
        return []

    # Convert events to role messages
    msgs = [{"role": role_for_sender(e["sender"], my_name), "content": e["text"]} for e in session]

    # Optional: drop leading message from me
    # while msgs and msgs[0]["role"] == "assistant":
    #     msgs = msgs[1:]

    # If first message is from me, prepend a dummy user prompt.
    if msgs and msgs[0]["role"] == "assistant":
        # Preserve assistant-initiated conversations by creating a dummy prompt.
        msgs = [{"role": "user", "content": "<CONVERSATION_START>"}] + msgs

    if not msgs:
        return []

    examples: List[Dict[str, Any]] = []

    # Create windows; overlapping windows are okay and often beneficial.
    # Step size can be tuned; start with half window overlap.
    step = max(1, max_messages_per_example // 2)

    for start in range(0, len(msgs), step):
        chunk = msgs[start:start + max_messages_per_example]
        if len(chunk) < 2:
            continue

        # Require at least one assistant turn in the chunk
        if not any(m["role"] == "assistant" for m in chunk):
            continue

        out_msgs = []
        if include_system:
            out_msgs.append({"role": "system", "content": system_text})
        out_msgs.extend(chunk)

        examples.append({"messages": out_msgs})

        if start + max_messages_per_example >= len(msgs):
            break

    return examples

def convert_one_thread_to_jsonl(
    input_json_path: str,
    output_jsonl_path: str,
    my_name: str,
    skip_if_has_keys: Optional[List[str]] = None,
    merge_window_seconds: int = 3600,        # 1 hour, as requested
    session_gap_seconds: int = 6 * 3600,     # optional; adjust or set very large to effectively disable
    max_messages_per_example: int = 24,
    redact_urls: bool = True,
) -> Dict[str, int]:
    skip_if_has_keys = skip_if_has_keys or ["call_duration"]

    thread = json.loads(Path(input_json_path).read_text(encoding="utf-8"))
    raw = thread.get("messages", [])

    # Sort oldest -> newest (Messenger exports are often newest-first)
    raw = sorted(raw, key=lambda m: int(m.get("timestamp_ms", 0)))

    # Normalize + filter
    events = []
    for m in raw:
        e = normalize_message(m, skip_if_has_keys=skip_if_has_keys, redact_urls=redact_urls)
        if e is not None:
            events.append(e)

    # Merge consecutive messages from same sender within 1 hour
    events_merged = merge_consecutive_by_sender(
        events,
        merge_window_seconds=merge_window_seconds,
        join_with="<MSG_BREAK>",
    )

    # Split into sessions to avoid huge examples
    sessions = split_into_sessions(events_merged, session_gap_seconds=session_gap_seconds)

    # Build training examples
    examples = []
    for sess in sessions:
        examples.extend(
            build_chat_examples_from_session(
                sess,
                my_name=my_name,
                max_messages_per_example=max_messages_per_example,
                include_system=True,
                system_text="You are Soham Pardeshi, a kind, helpful person. You are talking to your friend.",
            )
        )

    out_path = Path(output_jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return {
        "raw_messages": len(raw),
        "kept_events": len(events),
        "merged_events": len(events_merged),
        "sessions": len(sessions),
        "examples_written": len(examples),
    }

def convert_directory(
    input_dir: str,
    output_dir: str,
    my_name: str,
    skip_if_has_keys: Optional[List[str]] = None,
    merge_window_seconds: int = 3 * 60 * 60,
    session_gap_seconds: int = 48 * 60 * 60,
    max_messages_per_example: int = 48,
    redact_urls: bool = True,
    recursive: bool = True,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """
    For every file named message_???.json in input_dir, produce thread_???.jsonl in output_dir.

    - If recursive=True, searches subdirectories.
    - If overwrite=False, skips outputs that already exist.
    - Returns summary stats.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists() or not in_dir.is_dir():
        raise ValueError(f"Input dir does not exist or is not a directory: {in_dir}")

    # Choose globbing strategy
    paths = list(in_dir.rglob("message_*.json") if recursive else in_dir.glob("message_*.json"))
    paths = sorted(paths)

    total_files = 0
    converted_files = 0
    skipped_files = 0
    error_files = 0

    totals = {
        "raw_messages": 0,
        "kept_events": 0,
        "merged_events": 0,
        "sessions": 0,
        "examples_written": 0,
    }

    per_file: List[Dict[str, Any]] = []

    for p in paths:
        m = MESSAGE_FILE_RE.match(p.name)
        if not m:
            # Matches message_*.json but not strictly numeric; skip to be safe
            skipped_files += 1
            continue

        idx = m.group(1)  # numeric suffix
        out_path = out_dir / f"thread_{idx}.jsonl"

        if out_path.exists() and not overwrite:
            skipped_files += 1
            continue

        total_files += 1
        try:
            stats = convert_one_thread_to_jsonl(
                input_json_path=str(p),
                output_jsonl_path=str(out_path),
                my_name=my_name,
                skip_if_has_keys=skip_if_has_keys,
                merge_window_seconds=merge_window_seconds,
                session_gap_seconds=session_gap_seconds,
                max_messages_per_example=max_messages_per_example,
                redact_urls=redact_urls,
            )
            converted_files += 1

            # accumulate totals
            for k in totals:
                totals[k] += int(stats.get(k, 0))

            per_file.append({
                "input": str(p),
                "output": str(out_path),
                **stats
            })

            print(f"[OK] {p.name} -> {out_path.name} | {stats}")

        except Exception as e:
            error_files += 1
            print(f"[ERR] Failed on {p}: {e}")

    return {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "recursive": recursive,
        "overwrite": overwrite,
        "files_found": len(paths),
        "files_matched_pattern": total_files,
        "files_converted": converted_files,
        "files_skipped": skipped_files,
        "files_errors": error_files,
        "totals": totals,
        "per_file": per_file,  # you can remove this if it’s too verbose
    }


### Single example

# stats = convert_one_thread_to_jsonl(
#     input_json_path="data/raw/message_1.json",
#     output_jsonl_path="data/cleaned/thread_1.jsonl",
#     my_name="Soham Pardeshi",
#     skip_if_has_keys=["call_duration"],   # you can add more keys here
#     merge_window_seconds=3 * 60 * 60,     # merged 3 hours
#     session_gap_seconds=48 * 60 * 60,     # messages that are 2 days older are separate
#     max_messages_per_example=48,
#     redact_urls=True,
# )
# print(stats)


### Directory of examples

summary = convert_directory(
    input_dir="data/raw",
    output_dir="data/cleaned",
    my_name="Soham Pardeshi",
    skip_if_has_keys=["call_duration"],
    merge_window_seconds=3 * 60 * 60,
    session_gap_seconds=48 * 60 * 60,
    max_messages_per_example=48,
    redact_urls=True,
    recursive=True,
    overwrite=True,   # set False if you want to skip already-existing outputs
)
print("SUMMARY:", summary["totals"])
print("FILES:", {
    "found": summary["files_found"],
    "converted": summary["files_converted"],
    "skipped": summary["files_skipped"],
    "errors": summary["files_errors"],
})
