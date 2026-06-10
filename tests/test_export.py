from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from convo_tools._export import graph_to_gexf


def _graph() -> dict:
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
        "edges_mentions": {("msg::1", "entity::PERSON::alice"), ("msg::1", "entity::ORG::openai")},
        "edges_cooc": {("entity::PERSON::alice", "entity::ORG::openai")},
        "edges_keywords": [("msg::1", "kw::hello", 0.5)],
    }


def test_gexf_output(tmp_path: Path) -> None:
    out = tmp_path / "test.gexf"
    graph_to_gexf(_graph(), out)

    assert out.exists()
    content = out.read_text()
    assert "node" in content and "edge" in content
    # Count node/edge occurrences as a simple XML sanity check
    assert content.count("<node ") + content.count("<node\n") >= 5
    assert content.count("<edge ") + content.count("<edge\n") >= 5


def test_gexf_no_nodes(tmp_path: Path) -> None:
    g = {
        "nodes": {},
        "edges_contains": set(), "edges_replies_to": set(),
        "edges_mentions": set(), "edges_cooc": set(), "edges_keywords": [],
    }
    out = tmp_path / "empty.gexf"
    graph_to_gexf(g, out)
    content = out.read_text()
    # Empty graph should have no <node> elements (closing </nodes> or self-closing <nodes/> is fine)
    assert "<node>" not in content
