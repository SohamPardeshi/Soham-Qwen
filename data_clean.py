import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

URL_RE = re.compile(r"https?://\S+")
# Matches message_<thread>_<idx>.json created by your new copy/rename step
NEW_MESSAGE_FILE_RE = re.compile(r"^message_(.+)_(\d+)\.json$", re.IGNORECASE)


def has_any_key(d: Dict[str, Any], keys: List[str]) -> bool:
    return any(k in d for k in keys)


def looks_like_call_or_system_text(text: str) -> bool:
    t = text.lower().strip()
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
      { sender, ts_ms, text }
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

    text = " ".join(text.split()).strip()
    if not text:
        return None

    words = len(text.split())

    return {
        "sender": msg.get("sender_name", ""),
        "ts_ms": int(msg.get("timestamp_ms", 0)),
        "text": text,
        "word_count": words,
    }


def merge_consecutive_by_sender(
    events: List[Dict[str, Any]],
    merge_window_seconds: int = 3600,
    join_with: str = "\n",
) -> List[Dict[str, Any]]:
    if not events:
        return []

    merged: List[Dict[str, Any]] = []
    window_ms = merge_window_seconds * 1000

    cur = dict(events[0])
    for nxt in events[1:]:
        same_sender = (nxt["sender"] == cur["sender"])
        gap = nxt["ts_ms"] - cur["ts_ms"]

        if same_sender and gap <= window_ms:
            cur["text"] = cur["text"] + join_with + nxt["text"]
            cur["ts_ms"] = nxt["ts_ms"]
            cur["word_count"] += nxt.get("word_count", 0)
        else:
            merged.append(cur)
            cur = dict(nxt)

    merged.append(cur)
    return merged


def split_into_sessions(
    events: List[Dict[str, Any]],
    session_gap_seconds: int = 6 * 3600,
) -> List[List[Dict[str, Any]]]:
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
    if not session:
        return []

    msgs = [{"role": role_for_sender(e["sender"], my_name), "content": e["text"]} for e in session]

    if msgs and msgs[0]["role"] == "assistant":
        msgs = [{"role": "user", "content": "<CONVERSATION_START>"}] + msgs

    if not msgs:
        return []

    examples: List[Dict[str, Any]] = []
    step = max(1, max_messages_per_example // 2)

    for start in range(0, len(msgs), step):
        chunk = msgs[start:start + max_messages_per_example]
        if len(chunk) < 2:
            continue

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
    merge_window_seconds: int = 3600,
    session_gap_seconds: int = 6 * 3600,
    max_messages_per_example: int = 24,
    redact_urls: bool = True,
) -> Dict[str, int]:
    skip_if_has_keys = skip_if_has_keys or ["call_duration"]

    thread = json.loads(Path(input_json_path).read_text(encoding="utf-8"))
    raw = thread.get("messages", [])
    raw = sorted(raw, key=lambda m: int(m.get("timestamp_ms", 0)))

    events: List[Dict[str, Any]] = []
    for m in raw:
        e = normalize_message(m, skip_if_has_keys=skip_if_has_keys, redact_urls=redact_urls)
        if e is not None:
            events.append(e)

    events_merged = merge_consecutive_by_sender(
        events,
        merge_window_seconds=merge_window_seconds,
        join_with="<MSG_BREAK>",
    )

    sessions = split_into_sessions(events_merged, session_gap_seconds=session_gap_seconds)

    examples: List[Dict[str, Any]] = []
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

    total_words = sum(e.get("word_count", 0) for e in events_merged)

    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return {
        "raw_messages": len(raw),
        "kept_events": len(events),
        "merged_events": len(events_merged),
        "sessions": len(sessions),
        "examples_written": len(examples),
        "total_words": total_words,
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
    For every file named message_<thread>_<idx>.json in input_dir, produce
    thread_<thread>_<idx>.jsonl in output_dir.

    This matches the new flattened/copy output structure like:
      message_jennyliang_10208450689498493_3.json
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_dir.exists() or not in_dir.is_dir():
        raise ValueError(f"Input dir does not exist or is not a directory: {in_dir}")

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
        "total_words": 0,
    }

    per_file: List[Dict[str, Any]] = []

    for p in paths:
        m = NEW_MESSAGE_FILE_RE.match(p.name)
        if not m:
            skipped_files += 1
            continue

        thread_id = m.group(1)  # everything between first "message_" and last "_<digits>.json"
        idx = m.group(2)

        out_path = out_dir / f"thread_{thread_id}_{idx}.jsonl"

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
        "per_file": per_file,
    }


# Directory run example
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
    overwrite=True,
)
print("SUMMARY:", summary["totals"])
print("FILES:", {
    "found": summary["files_found"],
    "converted": summary["files_converted"],
    "skipped": summary["files_skipped"],
    "errors": summary["files_errors"],
})
