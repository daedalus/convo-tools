from __future__ import annotations

import argparse
from pathlib import Path

from convo_tools._depth import run_depth
from convo_tools._graph_db import GraphDB


def _graph(reply_edges: set) -> dict:
    return {
        "nodes": {
            "conv::c1": {"label": "Conversation"},
            "conv::c2": {"label": "Conversation"},
            "msg::1": {"label": "Message", "role": "user", "text": "First"},
            "msg::2": {"label": "Message", "role": "assistant", "text": "Reply"},
            "msg::3": {"label": "Message", "role": "user", "text": "Follow-up"},
            "msg::4": {"label": "Message", "role": "assistant", "text": "Deep reply"},
            "msg::5": {"label": "Message", "role": "user", "text": "Orphan"},
        },
        "edges_contains": {
            ("conv::c1", "msg::1"), ("conv::c1", "msg::2"), ("conv::c1", "msg::3"), ("conv::c1", "msg::4"),
            ("conv::c2", "msg::5"),
        },
        "edges_replies_to": reply_edges,
        "edges_mentions": set(),
        "edges_cooc": set(),
        "edges_keywords": [],
    }


def test_depth_basic(tmp_path: Path, capsys) -> None:
    g = _graph({("msg::1", "msg::2"), ("msg::2", "msg::3"), ("msg::3", "msg::4")})
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(g)
    db.close()

    args = argparse.Namespace(top=20, output=None)
    run_depth(db_path, args)
    captured = capsys.readouterr().out
    assert "depth" in captured


def test_depth_no_reply_edges(tmp_path: Path, capsys) -> None:
    g = _graph(set())
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(g)
    db.close()

    args = argparse.Namespace(top=20, output=None)
    run_depth(db_path, args)
    captured = capsys.readouterr().out
    assert "No reply edges" in captured


def test_depth_csv(tmp_path: Path, capsys) -> None:
    g = _graph({("msg::1", "msg::2")})
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(g)
    db.close()
    out_csv = tmp_path / "out.csv"

    args = argparse.Namespace(top=20, output=out_csv)
    run_depth(db_path, args)
    assert out_csv.exists()
    assert out_csv.read_text().startswith("conv_id")
