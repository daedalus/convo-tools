from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from convo_tools._timeline import run_timeline


def _graph() -> dict:
    return {
        "nodes": {
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
            "entity::ORG::openai": {"label": "Entity", "name": "openai", "entity_type": "ORG"},
        },
        "edges_mentions": {
            ("msg::1", "entity::PERSON::alice"),
            ("msg::2", "entity::ORG::openai"),
            ("msg::3", "entity::PERSON::alice"),
        },
        "edges_contains": {
            ("conv::c1", "msg::1"), ("conv::c1", "msg::2"), ("conv::c1", "msg::3"),
        },
        "edges_replies_to": set(),
        "edges_cooc": set(),
        "edges_keywords": [],
    }


def _messages(ts: float) -> list[dict]:
    return [
        {"id": "msg::1", "conversation_id": "conv::c1", "role": "user", "text": "hi", "create_time": ts},
        {"id": "msg::2", "conversation_id": "conv::c1", "role": "assistant", "text": "hello", "create_time": ts + 100},
        {"id": "msg::3", "conversation_id": "conv::c1", "role": "user", "text": "more", "create_time": ts + 200},
    ]


def test_timeline_basic(tmp_path: Path, capsys) -> None:
    g = _graph()
    pkl_g = tmp_path / "graph.pkl"
    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump(g, f)
    with open(pkl_m, "wb") as f:
        pickle.dump(_messages(1_000_000.0), f)

    args = argparse.Namespace(graph=pkl_g, messages=pkl_m, top=5, freq="year", output=None)
    run_timeline(args)
    out = capsys.readouterr().out
    assert "alice" in out


def test_timeline_csv(tmp_path: Path) -> None:
    g = _graph()
    pkl_g = tmp_path / "graph.pkl"
    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump(g, f)
    with open(pkl_m, "wb") as f:
        pickle.dump(_messages(1_000_000.0), f)
    out_csv = tmp_path / "out.csv"

    args = argparse.Namespace(graph=pkl_g, messages=pkl_m, top=5, freq="month", output=out_csv)
    run_timeline(args)
    assert out_csv.exists()


def test_timeline_no_timestamps(tmp_path: Path, capsys) -> None:
    g = _graph()
    pkl_g = tmp_path / "graph.pkl"
    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump(g, f)
    with open(pkl_m, "wb") as f:
        pickle.dump(
            [{"id": "msg::1", "conversation_id": "conv::c1", "role": "user", "text": "hi"}],
            f,
        )

    args = argparse.Namespace(graph=pkl_g, messages=pkl_m, top=5, freq="month", output=None)
    run_timeline(args)
    out = capsys.readouterr().out
    assert "unknown" in out or "alice" in out
