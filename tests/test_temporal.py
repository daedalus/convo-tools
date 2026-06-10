from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from convo_tools._graph_db import GraphDB
from convo_tools._temporal import run_temporal


def _graph() -> dict:
    return {
        "nodes": {
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
            "entity::TOPIC::ai": {"label": "Entity", "name": "ai", "entity_type": "TOPIC"},
        },
        "edges_mentions": {
            ("msg::1", "entity::PERSON::alice"),
            ("msg::2", "entity::TOPIC::ai"),
            ("msg::3", "entity::PERSON::alice"),
        },
        "edges_contains": {("conv::c1", "msg::1"), ("conv::c1", "msg::2"), ("conv::c1", "msg::3")},
        "edges_replies_to": set(), "edges_cooc": set(), "edges_keywords": [],
    }


def _messages() -> list[dict]:
    return [
        {"id": "msg::1", "conversation_id": "conv::c1", "role": "user", "text": "hi", "create_time": 1_000_000.0},
        {"id": "msg::2", "conversation_id": "conv::c1", "role": "assistant", "text": "AI", "create_time": 1_010_000.0},
        {"id": "msg::3", "conversation_id": "conv::c1", "role": "user", "text": "more", "create_time": 2_000_000.0},
    ]


def test_temporal_basic(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(_graph())
    db.close()

    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_m, "wb") as f:
        pickle.dump(_messages(), f)

    args = argparse.Namespace(messages=pkl_m, window=30, top=20, co_activation=False, output=None)
    run_temporal(db_path, args)
    out = capsys.readouterr().out
    assert "alice" in out
    assert "Bucket activity" in out


def test_temporal_co_activation(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(_graph())
    db.close()

    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_m, "wb") as f:
        pickle.dump(_messages(), f)

    args = argparse.Namespace(messages=pkl_m, window=30, top=20, co_activation=True, output=None)
    run_temporal(db_path, args)
    out = capsys.readouterr().out
    assert "co-activating" in out


def test_temporal_csv(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(_graph())
    db.close()

    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_m, "wb") as f:
        pickle.dump(_messages(), f)
    out_csv = tmp_path / "out.csv"

    args = argparse.Namespace(messages=pkl_m, window=30, top=20, co_activation=False, output=out_csv)
    run_temporal(db_path, args)
    assert out_csv.exists()


def test_temporal_pseudo_time(tmp_path: Path, capsys) -> None:
    g = _graph()
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(g)
    db.close()

    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_m, "wb") as f:
        pickle.dump([
            {"id": "msg::1", "conversation_id": "conv::c1", "role": "user", "text": "hi"},
            {"id": "msg::2", "conversation_id": "conv::c1", "role": "assistant", "text": "AI"},
        ], f)

    args = argparse.Namespace(messages=pkl_m, window=30, top=20, co_activation=False, output=None)
    run_temporal(db_path, args)
    assert "pseudo-time" in capsys.readouterr().out
