from __future__ import annotations

from convo_tools._graph_db import GraphDB
from convo_tools._semantic import semantic_search


def _build_test_db(tmp_path) -> GraphDB:
    db = GraphDB(tmp_path / "test.db")
    db.add_graph_batch({
        "nodes": {
            "conv::c1": {"label": "Conversation"},
            "conv::c2": {"label": "Conversation"},
            "msg::m1": {"label": "Message", "role": "user", "text": "Tell me about Python performance"},
            "msg::m2": {"label": "Message", "role": "assistant", "text": "Python has excellent garbage collection"},
            "msg::m3": {"label": "Message", "role": "user", "text": "How does Rust compare to C++?"},
            "msg::m4": {"label": "Message", "role": "assistant", "text": "Rust is faster and safer than C++"},
            "entity::ORG::python": {"label": "Entity", "name": "python", "entity_type": "ORG"},
            "entity::LANG::rust": {"label": "Entity", "name": "rust", "entity_type": "LANG"},
            "entity::LANG::cpp": {"label": "Entity", "name": "c++", "entity_type": "LANG"},
            "kw::performance": {"label": "Keyword", "name": "performance"},
            "kw::garbage_collection": {"label": "Keyword", "name": "garbage collection"},
        },
        "edges_contains": {
            ("conv::c1", "msg::m1"), ("conv::c1", "msg::m2"),
            ("conv::c2", "msg::m3"), ("conv::c2", "msg::m4"),
        },
        "edges_mentions": {
            ("msg::m1", "entity::ORG::python"), ("msg::m2", "entity::ORG::python"),
            ("msg::m3", "entity::LANG::rust"), ("msg::m3", "entity::LANG::cpp"),
            ("msg::m4", "entity::LANG::rust"), ("msg::m4", "entity::LANG::cpp"),
        },
        "edges_replies_to": {("msg::m1", "msg::m2"), ("msg::m3", "msg::m4")},
        "edges_cooc": set(),
        "edges_keywords": [
            ("msg::m1", "kw::performance", 0.9),
            ("msg::m2", "kw::garbage_collection", 0.8),
        ],
    })
    return db


def test_semantic_search_entity_match(tmp_path) -> None:
    db = _build_test_db(tmp_path)
    results = semantic_search(db, "python", top=5)
    db.close()
    msg_ids = {r["id"] for r in results}
    assert "msg::m1" in msg_ids
    assert "msg::m2" in msg_ids


def test_semantic_search_keyword_match(tmp_path) -> None:
    db = _build_test_db(tmp_path)
    results = semantic_search(db, "performance", top=5, use_embeddings=False, use_derived=False)
    db.close()
    msg_ids = {r["id"] for r in results}
    assert "msg::m1" in msg_ids


def test_semantic_search_no_match(tmp_path) -> None:
    db = _build_test_db(tmp_path)
    results = semantic_search(db, "quantum physics", top=5, use_embeddings=False, use_derived=False)
    db.close()
    assert len(results) == 0


def test_semantic_search_entity_only(tmp_path) -> None:
    db = _build_test_db(tmp_path)
    results = semantic_search(db, "python", top=5, use_embeddings=False, use_derived=False, use_keywords=False)
    db.close()
    msg_ids = {r["id"] for r in results}
    assert "msg::m1" in msg_ids
    assert "msg::m2" in msg_ids


def test_semantic_search_result_structure(tmp_path) -> None:
    db = _build_test_db(tmp_path)
    results = semantic_search(db, "python", top=5)
    db.close()
    assert len(results) > 0
    r = results[0]
    assert "id" in r
    assert "score" in r
    assert "role" in r
    assert "text" in r
    assert "matched_entities" in r
    assert "matched_keywords" in r
    assert "source" in r
    assert r["score"] > 0


def test_semantic_search_top_limit(tmp_path) -> None:
    db = _build_test_db(tmp_path)
    results = semantic_search(db, "python", top=1)
    db.close()
    assert len(results) <= 1


def test_semantic_search_entity_neighborhood(tmp_path) -> None:
    from convo_tools._semantic import get_entity_neighborhood
    db = _build_test_db(tmp_path)
    neighborhood = get_entity_neighborhood(db, "entity::ORG::python", depth=1)
    db.close()
    assert neighborhood["entity"] is not None
    assert neighborhood["entity"]["id"] == "entity::ORG::python"
    assert len(neighborhood["direct_messages"]) > 0
