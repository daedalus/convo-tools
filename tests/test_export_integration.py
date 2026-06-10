from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from convo_tools._export import run_export
from convo_tools._graph_db import GraphDB


def _graph() -> dict:
    return {
        "nodes": {
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
        },
        "edges_mentions": {("msg::1", "entity::PERSON::alice")},
        "edges_cooc": set(),
        "edges_contains": {("conv::c1", "msg::1")},
        "edges_replies_to": {("msg::1", "msg::2")},
        "edges_keywords": [],
    }


def _db_factory(data: dict, db_path: Path) -> GraphDB:
    db = GraphDB(db_path)
    db.add_graph_batch(data)
    return db


def test_run_export_basic(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    _db_factory(_graph(), db_path)
    out = tmp_path / "test.gexf"

    run_export(db_path, argparse_namespace(out))
    assert out.exists()
    content = out.read_text()
    assert "alice" in content


def test_run_export_replies_to(tmp_path: Path) -> None:
    g = _graph()
    g["edges_replies_to"] = {("msg::1", "msg::2")}
    g["edges_mentions"] = set()
    g["nodes"] = {
        "msg::1": {"label": "Message", "text": "hello"},
        "msg::2": {"label": "Message", "text": "reply"},
    }
    db_path = tmp_path / "test.db"
    _db_factory(g, db_path)
    out = tmp_path / "test.gexf"

    run_export(db_path, argparse_namespace(out))
    content = out.read_text()
    assert "REPLIES_TO" in content


def test_run_export_empty_graph(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "empty.db"
    out = tmp_path / "test.gexf"

    run_export(db_path, argparse_namespace(out))
    content = out.read_text()
    assert "<node>" not in content


def argparse_namespace(out: Path):
    return type("Args", (), {"output": out})()
