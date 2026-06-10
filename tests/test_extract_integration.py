from __future__ import annotations

import json
import pickle
from pathlib import Path

from convo_tools._extract import run_extract


def _make_conversation_json(conversation_id: str, messages: list[dict]) -> dict:
    return {
        "conversation_id": conversation_id,
        "conversation_title": f"Chat {conversation_id}",
        "mapping": {
            f"node_{i}": {
                "parent": m.get("parent"),
                "message": {
                    "id": m["id"],
                    "content": {
                        "content_type": "text",
                        "parts": [m["text"]],
                    },
                    "author": {"role": m.get("role", "user")},
                    "create_time": m.get("create_time", 1000.0 + i),
                },
            }
            for i, m in enumerate(messages)
        },
    }


def test_run_extract_basic(tmp_path: Path) -> None:
    data = _make_conversation_json("c1", [
        {"id": "msg::1", "role": "user", "text": "Hello", "create_time": 1000.0},
        {"id": "msg::2", "role": "assistant", "text": "Hi there", "create_time": 1001.0},
    ])
    json_file = tmp_path / "conv_c1.json"
    json_file.write_text(json.dumps(data))

    out_pkl = tmp_path / "messages.pkl"

    run_extract(json_dir=tmp_path, pickle_path=out_pkl)

    assert out_pkl.exists()
    msgs = pickle.loads(out_pkl.read_bytes())
    assert len(msgs) == 2


def test_run_extract_dedup(tmp_path: Path) -> None:
    text = "duplicate text"
    data = _make_conversation_json("c1", [
        {"id": "msg::1", "role": "user", "text": text, "create_time": 1000.0},
    ])
    json_file = tmp_path / "conv_c1.json"
    json_file.write_text(json.dumps(data))

    data2 = _make_conversation_json("c2", [
        {"id": "msg::2", "role": "user", "text": text, "create_time": 1001.0},
    ])
    json_file2 = tmp_path / "conv_c2.json"
    json_file2.write_text(json.dumps(data2))

    out_pkl = tmp_path / "messages.pkl"
    run_extract(json_dir=tmp_path, pickle_path=out_pkl)

    msgs = pickle.loads(out_pkl.read_bytes())
    assert len(msgs) == 1


def test_run_extract_no_json_files(tmp_path: Path, capsys) -> None:
    out_pkl = tmp_path / "messages.pkl"
    run_extract(json_dir=tmp_path, pickle_path=out_pkl)
    out = capsys.readouterr().out
    assert "Found 0 conversation files" in out or "No JSON files" in out


def test_run_extract_malformed_json(tmp_path: Path, capsys) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("not valid json")
    out_pkl = tmp_path / "messages.pkl"

    run_extract(json_dir=tmp_path, pickle_path=out_pkl)

    assert out_pkl.exists()
