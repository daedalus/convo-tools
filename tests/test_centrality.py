from __future__ import annotations

import argparse
from pathlib import Path

from convo_tools._centrality import run_centrality
from convo_tools._graph_db import GraphDB


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
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(_graph())
    db.close()

    args = argparse.Namespace(top=20, samples=0, exact=True, output=None)
    run_centrality(db_path, args)
    captured = capsys.readouterr().out
    assert "centrality" in captured
    assert "openai" in captured


def test_centrality_csv(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(_graph())
    db.close()
    out_csv = tmp_path / "out.csv"

    args = argparse.Namespace(top=20, samples=0, exact=True, output=out_csv)
    run_centrality(db_path, args)
    assert out_csv.exists()
    assert "entity_id" in out_csv.read_text()


def test_centrality_too_small(tmp_path: Path, capsys) -> None:
    g = {
        "nodes": {"e1": {"label": "Entity", "name": "e1", "entity_type": "PERSON"}},
        "edges_contains": set(), "edges_replies_to": set(),
        "edges_mentions": set(), "edges_cooc": set(), "edges_keywords": [],
    }
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch(g)
    db.close()

    args = argparse.Namespace(top=20, samples=0, exact=True, output=None)
    run_centrality(db_path, args)
    assert "too small" in capsys.readouterr().out


def test_centrality_missing_nodes_key(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "test.db"
    db = GraphDB(db_path)
    db.add_graph_batch({"nodes": {}, "edges_cooc": set(),
                        "edges_contains": set(), "edges_replies_to": set(),
                        "edges_mentions": set(), "edges_keywords": []})
    db.close()

    args = argparse.Namespace(top=20, samples=0, exact=True, output=None)
    run_centrality(db_path, args)
    assert "too small" in capsys.readouterr().out
