from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from convo_tools._diff import run_diff


def _graph_a() -> dict:
    return {
        "nodes": {
            "conv::c1": {"label": "Conversation"},
            "msg::1": {"label": "Message", "role": "user", "text": "Hello"},
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
            "entity::ORG::openai": {"label": "Entity", "name": "openai", "entity_type": "ORG"},
            "kw::hello": {"label": "Keyword", "name": "hello"},
        },
        "edges_contains": {("conv::c1", "msg::1")},
        "edges_replies_to": set(),
        "edges_mentions": {("msg::1", "entity::PERSON::alice")},
        "edges_cooc": set(),
        "edges_keywords": [("msg::1", "kw::hello", 0.5)],
    }


def _graph_b() -> dict:
    return {
        "nodes": {
            "conv::c1": {"label": "Conversation"},
            "msg::1": {"label": "Message", "role": "user", "text": "Hello"},
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
            "entity::PERSON::bob": {"label": "Entity", "name": "bob", "entity_type": "PERSON"},
            "kw::hello": {"label": "Keyword", "name": "hello"},
        },
        "edges_contains": {("conv::c1", "msg::1")},
        "edges_replies_to": set(),
        "edges_mentions": {("msg::1", "entity::PERSON::alice"), ("msg::1", "entity::PERSON::bob")},
        "edges_cooc": set(),
        "edges_keywords": [("msg::1", "kw::hello", 0.5)],
    }


def test_diff_basic(tmp_path: Path, capsys) -> None:
    pkl_a = tmp_path / "a.pkl"
    pkl_b = tmp_path / "b.pkl"
    with open(pkl_a, "wb") as f:
        pickle.dump(_graph_a(), f)
    with open(pkl_b, "wb") as f:
        pickle.dump(_graph_b(), f)

    args = argparse.Namespace(left=pkl_a, right=pkl_b, left_label="left", right_label="right", top=30, output=None)
    run_diff(args)
    out = capsys.readouterr().out
    assert "left" in out
    assert "right" in out
    assert "alice" in out


def test_diff_csv(tmp_path: Path) -> None:
    pkl_a = tmp_path / "a.pkl"
    pkl_b = tmp_path / "b.pkl"
    with open(pkl_a, "wb") as f:
        pickle.dump(_graph_a(), f)
    with open(pkl_b, "wb") as f:
        pickle.dump(_graph_b(), f)
    out_csv = tmp_path / "out.csv"

    args = argparse.Namespace(left=pkl_a, right=pkl_b, left_label="left", right_label="right", top=30, output=out_csv)
    run_diff(args)
    assert out_csv.exists()


def test_diff_missing_nodes(tmp_path: Path, capsys) -> None:
    pkl_a = tmp_path / "a.pkl"
    pkl_b = tmp_path / "b.pkl"
    with open(pkl_a, "wb") as f:
        pickle.dump({"x": 1}, f)
    with open(pkl_b, "wb") as f:
        pickle.dump({"x": 1}, f)

    args = argparse.Namespace(left=pkl_a, right=pkl_b, left_label="left", right_label="right", top=30, output=None)
    run_diff(args)
    assert "missing" in capsys.readouterr().err
