from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from convo_tools._topics import run_topics


def _graph() -> dict:
    return {
        "nodes": {
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
            "entity::PERSON::bob": {"label": "Entity", "name": "bob", "entity_type": "PERSON"},
            "entity::ORG::openai": {"label": "Entity", "name": "openai", "entity_type": "ORG"},
            "entity::ORG::google": {"label": "Entity", "name": "google", "entity_type": "ORG"},
            "entity::TOPIC::ai": {"label": "Entity", "name": "ai", "entity_type": "TOPIC"},
            "kw::ml": {"label": "Keyword", "name": "ml"},
        },
        "edges_cooc": {
            ("entity::PERSON::alice", "entity::ORG::openai"),
            ("entity::PERSON::alice", "entity::TOPIC::ai"),
            ("entity::PERSON::bob", "entity::ORG::google"),
            ("entity::PERSON::bob", "entity::TOPIC::ai"),
            ("entity::ORG::openai", "entity::TOPIC::ai"),
            ("entity::ORG::google", "entity::TOPIC::ai"),
        },
        "edges_mentions": {
            ("msg::1", "entity::PERSON::alice"),
            ("msg::2", "entity::ORG::openai"),
            ("msg::2", "entity::TOPIC::ai"),
            ("msg::3", "entity::PERSON::bob"),
        },
        "edges_contains": {
            ("conv::c1", "msg::1"), ("conv::c1", "msg::2"), ("conv::c1", "msg::3"),
        },
        "edges_replies_to": set(),
        "edges_keywords": [("msg::2", "kw::ml", 0.7)],
    }


def test_topics_basic(tmp_path: Path, capsys) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(_graph(), f)

    args = argparse.Namespace(pickle_path=pkl, top=10, min_size=2, output=None)
    run_topics(args)
    out = capsys.readouterr().out
    assert "Cluster" in out or "cluster" in out or "community" in out


def test_topics_csv(tmp_path: Path) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(_graph(), f)
    out_csv = tmp_path / "out.csv"

    args = argparse.Namespace(pickle_path=pkl, top=10, min_size=2, output=out_csv)
    run_topics(args)
    assert out_csv.exists()


def test_topics_too_small(tmp_path: Path, capsys) -> None:
    g = {
        "nodes": {"e1": {"label": "Entity", "name": "e1", "entity_type": "PERSON"}},
        "edges_cooc": set(), "edges_mentions": set(), "edges_contains": set(),
        "edges_replies_to": set(), "edges_keywords": [],
    }
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(g, f)

    args = argparse.Namespace(pickle_path=pkl, top=10, min_size=3, output=None)
    run_topics(args)
    assert "too small" in capsys.readouterr().out
