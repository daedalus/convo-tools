from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from convo_tools._graph_db import GraphDB
from convo_tools._similarity import run_similarity


def _graph() -> dict:
    return {
        "nodes": {
            "conv::a": {"label": "Conversation"},
            "conv::b": {"label": "Conversation"},
            "msg::a1": {"label": "Message", "role": "user", "text": "Hello"},
            "msg::a2": {"label": "Message", "role": "assistant", "text": "World"},
            "msg::b1": {"label": "Message", "role": "user", "text": "Another"},
            "entity::TOPIC::ai": {"label": "Entity", "name": "ai", "entity_type": "TOPIC"},
            "entity::TOPIC::math": {"label": "Entity", "name": "math", "entity_type": "TOPIC"},
            "kw::hello": {"label": "Keyword", "name": "hello"},
        },
        "edges_mentions": {
            ("msg::a1", "entity::TOPIC::ai"),
            ("msg::a2", "entity::TOPIC::ai"),
            ("msg::b1", "entity::TOPIC::math"),
        },
        "edges_contains": {
            ("conv::a", "msg::a1"), ("conv::a", "msg::a2"),
            ("conv::b", "msg::b1"),
        },
        "edges_replies_to": set(),
        "edges_cooc": set(),
        "edges_keywords": [("msg::a1", "kw::hello", 0.5)],
    }


def _messages() -> list[dict]:
    return [
        {"id": "msg::a1", "conversation_id": "conv::a", "role": "user", "text": "Hello"},
        {"id": "msg::a2", "conversation_id": "conv::a", "role": "assistant", "text": "World"},
        {"id": "msg::b1", "conversation_id": "conv::b", "role": "user", "text": "Another"},
    ]


def test_similarity_basic(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(_graph())
    db.close()

    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_m, "wb") as f:
        pickle.dump(_messages(), f)

    args = argparse.Namespace(messages=pkl_m, top=20, threshold=0.0, all=False, output=None)
    run_similarity(db_path, args)
    out = capsys.readouterr().out
    assert "Jaccard" in out


def test_similarity_csv(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(_graph())
    db.close()

    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_m, "wb") as f:
        pickle.dump(_messages(), f)
    out_csv = tmp_path / "out.csv"

    args = argparse.Namespace(messages=pkl_m, top=20, threshold=0.0, all=True, output=out_csv)
    run_similarity(db_path, args)
    assert out_csv.exists()


def test_similarity_no_pairs(tmp_path: Path, capsys) -> None:
    g = {
        "nodes": {"conv::a": {"label": "Conversation"}, "msg::a1": {"label": "Message", "role": "user", "text": "Hi"}},
        "edges_mentions": set(),
        "edges_contains": {("conv::a", "msg::a1")},
        "edges_replies_to": set(), "edges_cooc": set(), "edges_keywords": [],
    }
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(g)
    db.close()

    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_m, "wb") as f:
        pickle.dump([{"id": "msg::a1", "conversation_id": "conv::a", "role": "user", "text": "Hi"}], f)

    args = argparse.Namespace(messages=pkl_m, top=20, threshold=0.5, all=False, output=None)
    run_similarity(db_path, args)
    assert "No similar" in capsys.readouterr().out
