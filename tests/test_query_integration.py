from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from convo_tools._graph_db import GraphDB
from convo_tools._query import (
    _build_llm_context,
    _call_llm,
    _entity_name,
    _search_entities,
    _msg_text,
    _msg_role,
    run_query,
)


def test_entity_name_non_dict(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    result = _entity_name(db, "foo::bar::hello")
    assert result == "hello"
    db.close()


def test_entity_name_dict(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.upsert_node("foo::bar::hello", label="Entity", name="override")
    result = _entity_name(db, "foo::bar::hello")
    assert result == "override"
    db.close()


def test_entity_name_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    assert _entity_name(db, "nonexistent") == "nonexistent"
    db.close()


def test_msg_text_non_dict(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    assert _msg_text(db, "msg::1") == ""
    db.close()


def test_msg_role_non_dict(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    assert _msg_role(db, "msg::1") == "?"
    db.close()


def test_msg_text_missing(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    assert _msg_text(db, "msg::x") == ""
    db.close()


def test_search_entities_match_name(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.upsert_node("entity::PERSON::alice", label="Entity", name="Alice", entity_type="PERSON")
    db.upsert_node("msg::1", label="Message")
    db.add_edge_mentions("msg::1", "entity::PERSON::alice")
    edges_mentions = db.get_edges_mentions()
    matched, msg_map = _search_entities(["alice"], db, edges_mentions)
    assert "entity::PERSON::alice" in matched
    assert msg_map["entity::PERSON::alice"] == {"msg::1"}
    db.close()


def test_search_entities_name_fallback_split(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.upsert_node("entity::PERSON::alice-smith", label="Entity")
    db.upsert_node("msg::1", label="Message")
    db.add_edge_mentions("msg::1", "entity::PERSON::alice-smith")
    edges_mentions = db.get_edges_mentions()
    matched, _ = _search_entities(["alice"], db, edges_mentions)
    assert "entity::PERSON::alice-smith" in matched
    db.close()


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
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(g)
    db.close()

    args = argparse.Namespace(query="alice", messages=_empty_messages(tmp_path), llm=False, top=10, max_context=50000, output=None)
    run_query(db_path, args)
    out = capsys.readouterr().out
    assert "alice" in out.lower()


def test_query_no_results(tmp_path: Path, capsys) -> None:
    g = {
        "nodes": {},
        "edges_contains": set(), "edges_mentions": set(),
        "edges_replies_to": set(), "edges_cooc": set(), "edges_keywords": [],
    }
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(g)
    db.close()

    pkl_m = _empty_messages(tmp_path)
    args = argparse.Namespace(query="nonexistent", messages=pkl_m, llm=False, top=10, max_context=50000, output=None)
    run_query(db_path, args)
    out = capsys.readouterr().out
    assert "No matching" in out


def test_query_empty_query(tmp_path: Path, capsys) -> None:
    g = {"nodes": {}, "edges_contains": set(), "edges_mentions": set(),
         "edges_replies_to": set(), "edges_cooc": set(), "edges_keywords": []}
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(g)
    db.close()

    pkl_m = _empty_messages(tmp_path)
    args = argparse.Namespace(query="", messages=pkl_m, llm=False, top=10, max_context=50000, output=None)
    run_query(db_path, args)
    assert "no query" in capsys.readouterr().err


def test_query_malformed_graph(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.close()

    pkl_m = _empty_messages(tmp_path)
    args = argparse.Namespace(query="test", messages=pkl_m, llm=False, top=10, max_context=50000, output=None)
    run_query(db_path, args)
    assert "No matching" in capsys.readouterr().out


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
    assert "OPENAI_API_KEY" in result


def test_call_llm_import_error() -> None:
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        with patch("builtins.__import__", side_effect=ImportError):
            result = _call_llm("test query", "some context")
    assert "install" in result


def test_call_llm_api_error() -> None:
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        mock_openai = MagicMock()
        mock_client = MagicMock()
        mock_openai.OpenAI.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API error")
        with patch.dict("sys.modules", {"openai": mock_openai}):
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
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(g)
    db.close()

    args = argparse.Namespace(query="sequ", messages=_empty_messages(tmp_path), llm=False, top=10, max_context=50000, output=None)
    run_query(db_path, args)
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
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(g)
    db.close()

    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_m, "wb") as f:
        pickle.dump([{"id": "msg::1", "conversation_id": "conv::c1", "role": "user", "text": "About Alice", "create_time": 1000.0}], f)

    with patch("convo_tools._query._call_llm", return_value="Alice is a person"):
        args = argparse.Namespace(query="alice", messages=pkl_m, llm=True, top=10, max_context=50000, output=None)
        run_query(db_path, args)

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
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(g)
    db.close()

    out_csv = tmp_path / "out.csv"

    args = argparse.Namespace(query="alice", messages=_empty_messages(tmp_path), llm=False, top=10, max_context=50000, output=out_csv)
    run_query(db_path, args)
    assert out_csv.exists()
    csv_content = out_csv.read_text()
    assert "msg::1" in csv_content
