from __future__ import annotations

import json
from pathlib import Path

from convo_tools._join import (
    _parse_timestamp,
    _ts_iso,
    _ts_gemini_str,
    _ts_gemini_ms,
    _sorted_children,
    parse_file,
)


def test_parse_timestamp_none() -> None:
    assert _parse_timestamp(None) is None


def test_parse_timestamp_float_seconds() -> None:
    ts = _parse_timestamp(1234567890.0)
    assert ts == 1234567890.0


def test_parse_timestamp_float_millis() -> None:
    ts = _parse_timestamp(1700000000000.0)
    assert ts == 1700000000.0


def test_parse_timestamp_int_seconds() -> None:
    ts = _parse_timestamp(1000000000)
    assert ts == 1000000000.0


def test_parse_timestamp_int_millis() -> None:
    ts = _parse_timestamp(1700000000000)
    assert ts == 1700000000.0


def test_parse_timestamp_iso_format() -> None:
    ts = _parse_timestamp("2024-01-15T10:30:00Z")
    assert ts is not None


def test_parse_timestamp_iso_with_tz() -> None:
    ts = _parse_timestamp("2024-01-15T10:30:00+00:00")
    assert ts is not None


def test_parse_timestamp_simple_date() -> None:
    ts = _parse_timestamp("2024-01-15 10:30:00")
    assert ts is not None


def test_parse_timestamp_bad_string() -> None:
    ts = _parse_timestamp("not a timestamp")
    assert ts is None


def test_ts_iso_with_value() -> None:
    result = _ts_iso(1000000000.0)
    assert "T" in result
    assert result.endswith("Z")


def test_ts_iso_none() -> None:
    result = _ts_iso(None)
    assert "T" in result
    assert result.endswith("Z")


def test_ts_gemini_str_with_value() -> None:
    result = _ts_gemini_str(1000000000.0)
    assert ":" in result


def test_ts_gemini_str_none() -> None:
    result = _ts_gemini_str(None)
    assert ":" in result


def test_ts_gemini_ms_with_value() -> None:
    result = _ts_gemini_ms(1000000000.0)
    assert isinstance(result, int)


def test_ts_gemini_ms_none() -> None:
    result = _ts_gemini_ms(None)
    assert isinstance(result, int)


def test_sorted_children(tmp_path: Path) -> None:
    mapping = {
        "root": {"parent": None, "children": ["child1", "child2"], "message": {"id": "root_msg", "create_time": 1000.0}},
        "child1": {"parent": "root", "children": [], "message": {"id": "c1", "create_time": 1001.0}},
        "child2": {"parent": "root", "children": [], "message": {"id": "c2", "create_time": 1002.0}},
    }
    result = _sorted_children(mapping, "root")
    assert len(result) >= 2


def test_sorted_children_no_message(tmp_path: Path) -> None:
    mapping = {
        "root": {"parent": None, "children": ["child1"]},
        "child1": {"parent": "root", "children": [], "message": {"id": "c1", "create_time": 1001.0}},
    }
    result = _sorted_children(mapping, "root")
    assert len(result) == 1


def test_sorted_children_no_create_time(tmp_path: Path) -> None:
    mapping = {
        "root": {"parent": None, "children": ["c1", "c2"]},
        "c1": {"parent": "root", "children": [], "message": {"id": "c1"}},
        "c2": {"parent": "root", "children": [], "message": {"id": "c2"}},
    }
    result = _sorted_children(mapping, "root")
    assert len(result) == 2


def test_parse_file_openai(tmp_path: Path) -> None:
    data = {
        "conversation_id": "conv1",
        "title": "Test Chat",
        "mapping": {
            "n1": {
                "message": {
                    "id": "m1",
                    "content": {"content_type": "text", "parts": ["Hello"]},
                    "author": {"role": "user"},
                    "create_time": 1000000000.0,
                },
                "parent": None,
            },
        },
    }
    p = tmp_path / "openai.json"
    p.write_text(json.dumps(data))
    result = parse_file(str(p))
    assert len(result) >= 1


def test_parse_file_anthropic(tmp_path: Path) -> None:
    data = {
        "uuid": "conv1",
        "name": "Test",
        "chat_messages": [
            {
                "uuid": "m1",
                "sender": "human",
                "text": "Hello",
                "created_at": "2024-01-15T10:30:00Z",
            },
        ],
    }
    p = tmp_path / "anthropic.json"
    p.write_text(json.dumps(data))
    result = parse_file(str(p))
    assert len(result) >= 1

