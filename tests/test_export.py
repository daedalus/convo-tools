from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from convo_tools._export import graph_to_gexf
from convo_tools._graph_db import GraphDB


def _populate_db(db_path: Path) -> GraphDB:
    data = {
        "nodes": {
            "conv::c1": {"label": "Conversation"},
            "msg::1": {"label": "Message", "role": "user", "text": "Hello"},
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
            "entity::ORG::openai": {"label": "Entity", "name": "openai", "entity_type": "ORG"},
            "kw::hello": {"label": "Keyword", "name": "hello"},
        },
        "edges_contains": {("conv::c1", "msg::1")},
        "edges_replies_to": set(),
        "edges_mentions": {("msg::1", "entity::PERSON::alice"), ("msg::1", "entity::ORG::openai")},
        "edges_cooc": {("entity::PERSON::alice", "entity::ORG::openai")},
        "edges_keywords": [("msg::1", "kw::hello", 0.5)],
    }
    db = GraphDB(db_path)
    db.add_graph_batch(data)
    return db


def test_gexf_output(tmp_path: Path) -> None:
    out = tmp_path / "test.gexf"
    db = _populate_db(tmp_path / "test.db")
    graph_to_gexf(db, out)
    db.close()

    assert out.exists()
    content = out.read_text()
    assert "node" in content and "edge" in content
    assert content.count("<node ") + content.count("<node\n") >= 5
    assert content.count("<edge ") + content.count("<edge\n") >= 5


def test_gexf_no_nodes(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"
    db = GraphDB(db_path)
    out = tmp_path / "empty.gexf"
    graph_to_gexf(db, out)
    db.close()
    content = out.read_text()
    assert "<node>" not in content
