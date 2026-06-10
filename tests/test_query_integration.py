from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from convo_tools._query import (
    _build_llm_context,
    _call_llm,
    _entity_name,
    _search_entities,
    _msg_text,
    _msg_role,
    run_query,
)


def test_entity_name_non_dict() -> None:
    nodes = {"foo::bar::hello": "not_a_dict"}
    assert _entity_name(nodes, "foo::bar::hello") == "hello"


def test_entity_name_dict() -> None:
    nodes = {"foo::bar::hello": {"name": "override"}}
    assert _entity_name(nodes, "foo::bar::hello") == "override"


def test_entity_name_missing() -> None:
    assert _entity_name({}, "nonexistent") == "nonexistent"


def test_msg_text_non_dict() -> None:
    nodes = {"msg::1": "not_a_dict"}
    assert _msg_text(nodes, "msg::1") == ""


def test_msg_role_non_dict() -> None:
    nodes = {"msg::1": "not_a_dict"}
    assert _msg_role(nodes, "msg::1") == "?"


def test_msg_text_missing() -> None:
    assert _msg_text({}, "msg::x") == ""


def test_search_entities_match_name(tmp_path: Path) -> None:
    nodes = {
        "entity::PERSON::alice": {"label": "Entity", "name": "Alice", "entity_type": "PERSON"},
    }
    edges_mentions = {("msg::1", "entity::PERSON::alice")}
    matched, msg_map = _search_entities(["alice"], nodes, edges_mentions)
    assert "entity::PERSON::alice" in matched
    assert msg_map["entity::PERSON::alice"] == {"msg::1"}


def test_search_entities_name_fallback_split(tmp_path: Path) -> None:
    """When node has no 'name' key, entity name falls back to last :: segment."""
    nodes = {
        "entity::PERSON::alice-smith": {"label": "Entity"},
    }
    edges_mentions = {("msg::1", "entity::PERSON::alice-smith")}
    matched, _ = _search_entities(["alice"], nodes, edges_mentions)
    assert "entity::PERSON::alice-smith" in matched


def _empty_messages(tmp_path: Path) -> Path:
    p = tmp_path / "empty.pkl"
    with open(p, "wb") as f:
        pickle.dump([], f)
    return p


def test_query_entity_match(tmp_path: Path, capsys) -> None:
    g = {
        "nodes": {
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
        },
        "edges_contains": {("conv::c1", "msg::1")},
        "edges_mentions": {("msg::1", "entity::PERSON::alice")},
        "edges_replies_to": set(), "edges_cooc": set(), "edges_keywords": [],
    }
    pkl_g = tmp_path / "graph.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump(g, f)

    args = argparse.Namespace(query="alice", graph=pkl_g, messages=_empty_messages(tmp_path), llm=False, top=10, max_context=50000, output=None)
    run_query(args)
    out = capsys.readouterr().out
    assert "alice" in out.lower()


def test_query_no_results(tmp_path: Path, capsys) -> None:
    g = {
        "nodes": {},
        "edges_contains": set(), "edges_mentions": set(),
        "edges_replies_to": set(), "edges_cooc": set(), "edges_keywords": [],
    }
    pkl_g = tmp_path / "graph.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump(g, f)

    pkl_m = _empty_messages(tmp_path)
    args = argparse.Namespace(query="nonexistent", graph=pkl_g, messages=pkl_m, llm=False, top=10, max_context=50000, output=None)
    run_query(args)
    out = capsys.readouterr().out
    assert "No matching" in out


def test_query_empty_query(tmp_path: Path, capsys) -> None:
    g = {"nodes": {}, "edges_contains": set(), "edges_mentions": set(),
         "edges_replies_to": set(), "edges_cooc": set(), "edges_keywords": []}
    pkl_g = tmp_path / "graph.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump(g, f)

    pkl_m = _empty_messages(tmp_path)
    args = argparse.Namespace(query="", graph=pkl_g, messages=pkl_m, llm=False, top=10, max_context=50000, output=None)
    run_query(args)
    assert "no query" in capsys.readouterr().err


def test_query_malformed_graph(tmp_path: Path, capsys) -> None:
    pkl_g = tmp_path / "graph.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump("not_a_dict", f)

    pkl_m = _empty_messages(tmp_path)
    args = argparse.Namespace(query="test", graph=pkl_g, messages=pkl_m, llm=False, top=10, max_context=50000, output=None)
    run_query(args)
    assert "Error" in capsys.readouterr().err


def test_build_llm_context() -> None:
    results = [
        {"msg_id": "msg::1", "score": 0.9, "role": "user", "text": "Hello", "entities": "alice", "keywords": "", "conversation_id": "conv::c1", "timestamp": 1000.0},
    ]
    ctx = _build_llm_context(results, max_chars=50000)
    assert "conv::c1" in ctx
    assert "Hello" in ctx


def test_build_llm_context_truncation() -> None:
    results = [
        {"msg_id": f"msg::{i}", "score": 0.9, "role": "user", "text": "A" * 1000, "entities": "e", "keywords": "", "conversation_id": f"conv::{i}", "timestamp": 1000.0 + i}
        for i in range(100)
    ]
    ctx = _build_llm_context(results, max_chars=500)
    assert len(ctx) <= 600


def test_call_llm_no_api_key() -> None:
    with patch.dict("os.environ", {}, clear=True):
        result = _call_llm("test query", "some context")
    assert "ANTHROPIC_API_KEY" in result


def test_call_llm_import_error() -> None:
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
        with patch("builtins.__import__", side_effect=ImportError):
            result = _call_llm("test query", "some context")
    assert "install" in result


def test_call_llm_api_error() -> None:
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
        mock_anthropic = MagicMock()
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")
        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            result = _call_llm("test query", "some context")
    assert "Error" in result or "API" in result


def test_query_keyword_matching(tmp_path: Path, capsys) -> None:
    g = {
        "nodes": {
            "kw::sequence": {"label": "Keyword", "name": "sequence"},
            "msg::1": {"label": "Message", "text": "the sequence is important"},
        },
        "edges_contains": {("conv::c1", "msg::1")},
        "edges_mentions": set(),
        "edges_replies_to": set(), "edges_cooc": set(),
        "edges_keywords": [("msg::1", "kw::sequence", 0.85)],
    }
    pkl_g = tmp_path / "graph.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump(g, f)

    args = argparse.Namespace(query="sequ", graph=pkl_g, messages=_empty_messages(tmp_path), llm=False, top=10, max_context=50000, output=None)
    run_query(args)
    out = capsys.readouterr().out
    assert "sequence" in out or "msg::1" in out


def test_query_llm_mode(tmp_path: Path, capsys) -> None:
    g = {
        "nodes": {
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
        },
        "edges_contains": {("conv::c1", "msg::1")},
        "edges_mentions": {("msg::1", "entity::PERSON::alice")},
        "edges_replies_to": set(), "edges_cooc": set(), "edges_keywords": [],
    }
    pkl_g = tmp_path / "graph.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump(g, f)
    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_m, "wb") as f:
        pickle.dump([{"id": "msg::1", "conversation_id": "conv::c1", "role": "user", "text": "About Alice", "create_time": 1000.0}], f)

    with patch("convo_tools._query._call_llm", return_value="Alice is a person"):
        args = argparse.Namespace(query="alice", graph=pkl_g, messages=pkl_m, llm=True, top=10, max_context=50000, output=None)
        run_query(args)

    out = capsys.readouterr().out
    assert "Alice" in out
    assert "LLM Response" in out


def test_query_csv_output(tmp_path: Path) -> None:
    g = {
        "nodes": {
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
        },
        "edges_contains": {("conv::c1", "msg::1")},
        "edges_mentions": {("msg::1", "entity::PERSON::alice")},
        "edges_replies_to": set(), "edges_cooc": set(), "edges_keywords": [],
    }
    pkl_g = tmp_path / "graph.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump(g, f)
    out_csv = tmp_path / "out.csv"

    args = argparse.Namespace(query="alice", graph=pkl_g, messages=_empty_messages(tmp_path), llm=False, top=10, max_context=50000, output=out_csv)
    run_query(args)
    assert out_csv.exists()
    csv_content = out_csv.read_text()
    assert "msg::1" in csv_content
