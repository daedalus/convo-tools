from __future__ import annotations

import argparse
import json
from pathlib import Path

from convo_tools._split import run_split


def test_split_conversations(tmp_path: Path) -> None:
    data = [
        {"conversation_id": "c1", "title": "First", "messages": [{"role": "user", "text": "hi"}]},
        {"conversation_id": "c2", "title": "Second", "messages": [{"role": "assistant", "text": "hello"}]},
    ]
    inp = tmp_path / "input.json"
    inp.write_text(json.dumps(data))
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    args = argparse.Namespace(input_file=str(inp), output_dir=str(out_dir))
    run_split(args)

    files = sorted(out_dir.iterdir())
    assert len(files) == 2
    assert all(f.suffix == ".json" for f in files)


def test_split_empty(tmp_path: Path) -> None:
    inp = tmp_path / "empty.json"
    inp.write_text("[]")
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    args = argparse.Namespace(input_file=str(inp), output_dir=str(out_dir))
    run_split(args)
    assert len(list(out_dir.iterdir())) == 0
