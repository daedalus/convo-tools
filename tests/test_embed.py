from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from convo_tools._embed import run_embed


def _graph() -> dict:
    return {
        "nodes": {
            "msg::m1": {"label": "Message", "text": "Hello world about AI"},
            "msg::m2": {"label": "Message", "text": "AI is interesting"},
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
            "entity::ORG::openai": {"label": "Entity", "name": "openai", "entity_type": "ORG"},
            "entity::TOPIC::ai": {"label": "Entity", "name": "ai", "entity_type": "TOPIC"},
            "kw::ai": {"label": "Keyword", "name": "ai"},
        },
        "edges_contains": {("conv::c1", "msg::m1"), ("conv::c1", "msg::m2")},
        "edges_mentions": {
            ("msg::m1", "entity::PERSON::alice"),
            ("msg::m1", "entity::ORG::openai"),
            ("msg::m2", "entity::TOPIC::ai"),
        },
        "edges_replies_to": set(),
        "edges_cooc": {
            ("entity::PERSON::alice", "entity::ORG::openai"),
        },
        "edges_keywords": [("msg::m1", "kw::ai", 0.8)],
    }


def test_embed_basic(tmp_path: Path, capsys) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(_graph(), f)

    args = argparse.Namespace(pickle_path=pkl, output=None, save=None, load=None, similar_to=None, top=5, embed_kwargs=True, dim=32, n_iter=5, all=False)
    run_embed(args)
    out = capsys.readouterr().out
    assert "embedding" in out.lower() or "similar" in out.lower() or "dimensionality" in out.lower()


def test_embed_csv(tmp_path: Path, capsys) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(_graph(), f)
    out_csv = tmp_path / "out.csv"

    args = argparse.Namespace(pickle_path=pkl, output=out_csv, save=None, load=None, similar_to=None, top=5, embed_kwargs=True, dim=32, n_iter=5, all=False)
    run_embed(args)
    # --output without --similar-to prints a message and does not create the file
    out = capsys.readouterr().out
    assert "--similar-to" in out or "not create" in out


def test_embed_save_load(tmp_path: Path) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(_graph(), f)
    emb = tmp_path / "emb.npz"

    args = argparse.Namespace(pickle_path=pkl, output=None, save=emb, load=None, similar_to=None, top=5, embed_kwargs=True, dim=32, n_iter=5, all=False)
    run_embed(args)
    assert emb.exists()

    # Load back
    args2 = argparse.Namespace(pickle_path=pkl, output=None, save=None, load=emb, similar_to=None, top=5, embed_kwargs=True, dim=32, n_iter=5, all=False)
    run_embed(args2)


def test_embed_similar_to(tmp_path: Path, capsys) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(_graph(), f)

    args = argparse.Namespace(pickle_path=pkl, output=None, save=None, load=None, similar_to="msg::m1", top=5, embed_kwargs=True, dim=32, n_iter=5, all=False)
    run_embed(args)
    out = capsys.readouterr().out
    assert "msg::m1" in out or "similar" in out.lower()


def test_embed_no_keywords(tmp_path: Path, capsys) -> None:
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(_graph(), f)

    args = argparse.Namespace(pickle_path=pkl, output=None, save=None, load=None, similar_to=None, top=5, embed_kwargs=False, dim=32, n_iter=5, all=False)
    run_embed(args)
    out = capsys.readouterr().out


def test_embed_small_graph(tmp_path: Path, capsys) -> None:
    g = {
        "nodes": {"msg::m1": {"label": "Message", "text": "hi"}},
        "edges_contains": {("conv::c1", "msg::m1")},
        "edges_mentions": set(),
        "edges_replies_to": set(), "edges_cooc": set(), "edges_keywords": [],
    }
    pkl = tmp_path / "graph.pkl"
    with open(pkl, "wb") as f:
        pickle.dump(g, f)

    args = argparse.Namespace(pickle_path=pkl, output=None, save=None, load=None, similar_to=None, top=5, embed_kwargs=True, dim=32, n_iter=5, all=False)
    run_embed(args)
