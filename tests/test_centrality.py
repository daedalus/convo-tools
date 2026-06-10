from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from convo_tools._centrality import run_centrality


def _graph() -> dict:
    return {
        "nodes": {
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
            "entity::PERSON::bob": {"label": "Entity", "name": "bob", "entity_type": "PERSON"},
            "entity::ORG::openai": {"label": "Entity", "name": "openai", "entity_type": "ORG"},
        },
        "edges_contains": set(),
        "edges_replies_to": set(),
        "edges_mentions": set(),
        "edges_cooc": {
            ("entity::PERSON::alice", "entity::ORG::openai"),
            ("entity::PERSON::bob", "entity::ORG::openai"),
        },
        "edges_keywords": [],
    }


def test_centrality_basic(tmp_path: Path, capsys) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(_graph(), f)

    args = argparse.Namespace(pickle_path=pkl, top=20, samples=0, exact=True, output=None)
    run_centrality(args)
    captured = capsys.readouterr().out
    assert "centrality" in captured
    assert "openai" in captured


def test_centrality_csv(tmp_path: Path) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(_graph(), f)
    out_csv = tmp_path / "out.csv"

    args = argparse.Namespace(pickle_path=pkl, top=20, samples=0, exact=True, output=out_csv)
    run_centrality(args)
    assert out_csv.exists()
    assert "entity_id" in out_csv.read_text()


def test_centrality_too_small(tmp_path: Path, capsys) -> None:
    g = {
        "nodes": {"e1": {"label": "Entity", "name": "e1", "entity_type": "PERSON"}},
        "edges_contains": set(), "edges_replies_to": set(),
        "edges_mentions": set(), "edges_cooc": set(), "edges_keywords": [],
    }
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(g, f)

    args = argparse.Namespace(pickle_path=pkl, top=20, samples=0, exact=True, output=None)
    run_centrality(args)
    assert "too small" in capsys.readouterr().out


def test_centrality_missing_nodes_key(tmp_path: Path, capsys) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"edges_cooc": set()}, f)

    args = argparse.Namespace(pickle_path=pkl, top=20, samples=0, exact=True, output=None)
    run_centrality(args)
    assert "missing 'nodes' key" in capsys.readouterr().err
