from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import pytest

from convo_tools._graph_db import GraphDB
from convo_tools._ingest import run_ingest

kuzu = pytest.importorskip("kuzu")


def _graph() -> dict:
    return {
        "nodes": {
            "msg::m1": {"label": "Message", "role": "user", "text": "Hello"},
            "msg::m2": {"label": "Message", "role": "assistant", "text": "Hi there"},
            "conv::c1": {"label": "Conversation"},
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
            "entity::ORG::openai": {"label": "Entity", "name": "openai", "entity_type": "ORG"},
            "kw::hello": {"label": "Keyword", "name": "hello"},
        },
        "edges_contains": {("conv::c1", "msg::m1"), ("conv::c1", "msg::m2")},
        "edges_mentions": {("msg::m1", "entity::PERSON::alice"), ("msg::m2", "entity::ORG::openai")},
        "edges_replies_to": {("msg::m1", "msg::m2")},
        "edges_cooc": {("entity::PERSON::alice", "entity::ORG::openai")},
        "edges_keywords": [("msg::m1", "kw::hello", 0.8)],
    }


def test_ingest_basic(tmp_path: Path, capsys) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(_graph(), f)
    db = tmp_path / "test_db"

    args = argparse.Namespace(pickle_path=pkl, db=None, kuzu_path=db, overwrite=False)
    run_ingest(args)
    assert db.exists()


def test_ingest_overwrite(tmp_path: Path) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(_graph(), f)
    db = tmp_path / "test_db2"

    args = argparse.Namespace(pickle_path=pkl, db=None, kuzu_path=db, overwrite=True)
    run_ingest(args)
    assert db.exists()
    run_ingest(args)
    assert db.exists()


def test_ingest_from_db(tmp_path: Path) -> None:
    graph_db_path = tmp_path / "knowledge_graph.db"
    gdb = GraphDB(graph_db_path)
    gdb.add_graph_batch(_graph())
    gdb.close()

    kuzu_path = tmp_path / "test_kuzu_db"
    args = argparse.Namespace(pickle_path=None, db=graph_db_path, kuzu_path=kuzu_path, overwrite=False)
    run_ingest(args)
    assert kuzu_path.exists()


def test_ingest_small(tmp_path: Path, capsys) -> None:
    g = {
        "nodes": {},
        "edges_contains": set(), "edges_mentions": set(),
        "edges_replies_to": set(), "edges_cooc": set(), "edges_keywords": [],
    }
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(g, f)
    db = tmp_path / "tiny_db"

    args = argparse.Namespace(pickle_path=pkl, db=None, kuzu_path=db, overwrite=False)
    run_ingest(args)
    assert db.exists()
