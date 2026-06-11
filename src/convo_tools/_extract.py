from __future__ import annotations

import json
import pickle
from typing import TYPE_CHECKING, Any

from langdetect import LangDetectException, detect

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


def _process_conversation(
    conversation: dict[str, Any],
    conversation_id: str,
    seen_hashes: set[str],
) -> list[dict[str, Any]]:
    messages = extract_messages(conversation)
    result: list[dict[str, Any]] = []
    for msg in messages:
        msg["conversation_id"] = conversation_id
        h = text_hash(msg["text"])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        msg["lang"] = detect_lang(msg["text"])
        result.append(msg)
    return result


def run_extract(json_path: Path, pickle_path: Path) -> None:
    all_messages: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()

    if json_path.is_file():
        print(f"Reading single file: '{json_path}'")
        with open(json_path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            print(f"JSON array with {len(data)} conversations")
            for conversation in data:
                conv_id = conversation.get("conversation_id", conversation.get("id", ""))
                msgs = _process_conversation(conversation, conv_id, seen_hashes)
                if msgs:
                    print(f"  {conv_id[:40]}: {len(msgs)} messages")
                all_messages.extend(msgs)
        elif isinstance(data, dict):
            conv_id = data.get("conversation_id", data.get("id", json_path.stem))
            msgs = _process_conversation(data, conv_id, seen_hashes)
            print(f"  {conv_id[:40]}: {len(msgs)} messages")
            all_messages.extend(msgs)
        else:
            print(f"ERROR: unexpected JSON type: {type(data)}")
            return

    elif json_path.is_dir():
        json_files = list(json_path.glob("*.json"))
        print(f"Found {len(json_files)} conversation files in '{json_path}/'")

        for file in json_files:
            try:
                with open(file, encoding="utf-8") as f:
                    conversation = json.load(f)

                conversation_id = file.stem
                msgs = _process_conversation(conversation, conversation_id, seen_hashes)
                print(
                    f"  {file.name}: {len(extract_messages(conversation))} messages "
                    f"({len(extract_messages(conversation)) - len(msgs)} deduped)"
                )
                all_messages.extend(msgs)

            except Exception as e:
                print(f"ERROR {file}: {e}")

    else:
        print(f"ERROR: path does not exist: {json_path}")
        return

    with open(pickle_path, "wb") as f:
        pickle.dump(all_messages, f)

    print(f"\nSaved {len(all_messages)} deduplicated messages to {pickle_path}")
