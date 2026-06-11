from __future__ import annotations

import json
import pickle
from typing import TYPE_CHECKING, Any

from langdetect import detect, LangDetectException

from convo_tools._util import text_hash

if TYPE_CHECKING:
    from pathlib import Path


def detect_lang(text: str) -> str:
    try:
        return detect(text[:5000])
    except LangDetectException:
        return "unknown"


def extract_messages(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    mapping: dict[str, Any] = conversation.get("mapping", {})

    for node_id, node in mapping.items():
        message = node.get("message")
        if not message:
            continue

        role = message.get("author", {}).get("role", "unknown")
        content = message.get("content", {})

        if content.get("content_type") != "text":
            continue

        text = "\n".join(content.get("parts", []))
        create_time = message.get("create_time")
        messages.append(
            {
                "id": node_id,
                "role": role,
                "text": text,
                "parent": node.get("parent"),
                "create_time": create_time,
            }
        )

    return messages


def run_extract(json_dir: Path, pickle_path: Path) -> None:
    all_messages: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    json_files = list(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} conversation files in '{json_dir}/'")

    for file in json_files:
        try:
            with open(file, encoding="utf-8") as f:
                conversation = json.load(f)

            conversation_id = file.stem
            messages = extract_messages(conversation)

            kept = 0
            for msg in messages:
                msg["conversation_id"] = conversation_id
                h = text_hash(msg["text"])
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                msg["lang"] = detect_lang(msg["text"])
                all_messages.append(msg)
                kept += 1

            print(
                f"  {file.name}: {len(messages)} messages ({len(messages) - kept} deduped)"
            )

        except Exception as e:
            print(f"ERROR {file}: {e}")

    with open(pickle_path, "wb") as f:
        pickle.dump(all_messages, f)

    print(f"\nSaved {len(all_messages)} deduplicated messages to {pickle_path}")
