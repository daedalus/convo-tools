from __future__ import annotations

import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from convo_tools._export import run_export


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


def test_run_export_basic(tmp_path: Path) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(_graph(), f)
    out = tmp_path / "test.gexf"

    run_export(argparse_namespace(pkl, out))
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
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(g, f)
    out = tmp_path / "test.gexf"

    run_export(argparse_namespace(pkl, out))
    content = out.read_text()
    assert "REPLIES_TO" in content


def test_run_export_malformed(tmp_path: Path, capsys) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump("not_a_graph", f)
    out = tmp_path / "test.gexf"

    run_export(argparse_namespace(pkl, out))
    assert "Error" in capsys.readouterr().out
    assert not out.exists()


def argparse_namespace(pkl: Path, out: Path):
    return type("Args", (), {"pickle_path": pkl, "output": out})()
