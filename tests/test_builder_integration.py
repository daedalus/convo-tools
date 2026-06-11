from __future__ import annotations

import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from convo_tools._builder import build_graph_to_db, run_graph
from convo_tools._graph_db import GraphDB


def _make_mock_doc(entities: list[tuple[str, str]]):
    doc = MagicMock()
    doc.ents = []
    for text, label in entities:
        ent = MagicMock()
        ent.text = text
        ent.label_ = label
        doc.ents.append(ent)
    return doc


def _msg(text: str, **kw):
    return {
        "id": kw.get("id", "m1"),
        "conversation_id": kw.get("conv", "c1"),
        "role": kw.get("role", "user"),
        "text": text,
        "parent": kw.get("parent", None),
        "create_time": kw.get("create_time", 1000.0),
    }


def test_build_graph_with_entities(tmp_path: Path) -> None:
    messages = [
        _msg("Alice works at OpenAI", id="m1"),
        _msg("Bob is from Google", id="m2"),
    ]

    def mock_nlp_side(text: str):
        entities = []
        if "Alice" in text:
            entities.append(("Alice", "PERSON"))
        if "OpenAI" in text:
            entities.append(("OpenAI", "ORG"))
        if "Bob" in text:
            entities.append(("Bob", "PERSON"))
        if "Google" in text:
            entities.append(("Google", "ORG"))
        return _make_mock_doc(entities)

    mock_nlp = MagicMock()
    mock_nlp.side_effect = mock_nlp_side

    with patch("spacy.load", return_value=mock_nlp):
        with patch("sklearn.feature_extraction.text.TfidfVectorizer") as mock_tfidf:
            mock_v = MagicMock()
            mock_tfidf.return_value = mock_v
            mock_v.fit_transform.return_value = MagicMock()
            db = GraphDB(tmp_path / "test.db")
            build_graph_to_db(messages, db, debug=True)

    assert db.get_node("entity::PERSON::alice") is not None
    assert db.get_node("entity::PERSON::bob") is not None
    assert db.get_node("entity::ORG::openai") is not None
    assert db.get_node("entity::ORG::google") is not None

    edges_mentions = set(db.get_edges_mentions())
    assert ("m1", "entity::PERSON::alice") in edges_mentions
    assert ("m1", "entity::ORG::openai") in edges_mentions
    assert ("m2", "entity::PERSON::bob") in edges_mentions
    assert ("m2", "entity::ORG::google") in edges_mentions

    edges_cooc = set((a, b) for a, b, _ in db.get_edges_cooc())
    assert ("entity::ORG::openai", "entity::PERSON::alice") in edges_cooc or \
           ("entity::PERSON::alice", "entity::ORG::openai") in edges_cooc
    assert ("entity::ORG::google", "entity::PERSON::bob") in edges_cooc or \
           ("entity::PERSON::bob", "entity::ORG::google") in edges_cooc
    db.close()


def test_build_graph_cooc_two_entities(tmp_path: Path) -> None:
    messages = [_msg("Alice Bob chat", id="m1")]

    def mock_nlp_side(text: str):
        return _make_mock_doc([("Alice", "PERSON"), ("Bob", "PERSON")])

    mock_nlp = MagicMock()
    mock_nlp.side_effect = mock_nlp_side

    with patch("spacy.load", return_value=mock_nlp):
        with patch("sklearn.feature_extraction.text.TfidfVectorizer") as mock_tfidf:
            mock_v = MagicMock()
            mock_tfidf.return_value = mock_v
            mock_v.fit_transform.return_value = MagicMock()
            db = GraphDB(tmp_path / "test.db")
            build_graph_to_db(messages, db)

    edges_cooc = db.get_edges_cooc()
    assert len(edges_cooc) == 1
    pair = edges_cooc[0]
    assert "alice" in pair[0] and "bob" in pair[1]
    db.close()


def test_build_graph_empty_messages(tmp_path: Path) -> None:
    with patch("spacy.load"):
        with patch("sklearn.feature_extraction.text.TfidfVectorizer"):
            db = GraphDB(tmp_path / "test.db")
            build_graph_to_db([], db)

    assert len(db.get_all_nodes_by_label("Conversation")) == 0
    assert len(db.get_all_nodes_by_label("Message")) == 0
    assert db.get_edges_contains() == []
    assert db.get_edges_replies_to() == []
    assert db.get_edges_mentions() == []
    assert db.get_edges_cooc() == []
    assert db.get_edges_keywords() == []
    db.close()


def test_build_graph_repeated_keyword(tmp_path: Path) -> None:
    messages = [
        _msg("Alice loves apples", id="m1"),
        _msg("Bob loves apples too", id="m2"),
    ]

    def mock_nlp_side(text: str):
        ents = []
        if "Alice" in text:
            ents.append(("Alice", "PERSON"))
        if "Bob" in text:
            ents.append(("Bob", "PERSON"))
        return _make_mock_doc(ents)

    mock_nlp = MagicMock()
    mock_nlp.side_effect = mock_nlp_side

    with patch("spacy.load", return_value=mock_nlp):
        with patch("sklearn.feature_extraction.text.TfidfVectorizer") as mock_tfidf_cls:
            instance = MagicMock()
            mock_tfidf_cls.return_value = instance
            instance.get_feature_names_out.return_value = ["aple", "love"]
            from scipy.sparse import csr_matrix
            import numpy as np
            mat = csr_matrix(np.array([[0.8, 0.6], [0.7, 0.5]]))
            instance.fit_transform.return_value = mat

            db = GraphDB(tmp_path / "test.db")
            build_graph_to_db(messages, db)

    assert db.get_node("keyword::aple") is not None
    assert db.get_node("keyword::love") is not None
    kw_edges = db.get_edges_keywords()
    m1_kws = [(m, k, w) for (m, k, w) in kw_edges if m == "m1"]
    m2_kws = [(m, k, w) for (m, k, w) in kw_edges if m == "m2"]
    assert len(m1_kws) >= 1
    assert len(m2_kws) >= 1
    db.close()


def test_run_graph_fresh(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    messages = [_msg("Hello", id="m1")]
    msgs_pkl = tmp_path / "messages.pkl"
    with open(msgs_pkl, "wb") as f:
        pickle.dump(messages, f)

    def mock_nlp_side(text: str):
        return _make_mock_doc([])

    mock_nlp = MagicMock()
    mock_nlp.side_effect = mock_nlp_side

    with patch("spacy.load", return_value=mock_nlp):
        with patch("sklearn.feature_extraction.text.TfidfVectorizer") as mock_tfidf:
            mock_v = MagicMock()
            mock_tfidf.return_value = mock_v
            mock_v.fit_transform.return_value = MagicMock()
            run_graph(pickle_path=msgs_pkl, db_path=tmp_path / "knowledge_graph.db", export_pickle=True)

    assert (tmp_path / "knowledge_graph.pkl").exists()
    loaded = pickle.loads((tmp_path / "knowledge_graph.pkl").read_bytes())
    assert "nodes" in loaded


def test_run_graph_incremental(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "knowledge_graph.db"
    msg1 = _msg("Hello", id="m1")
    msg2 = _msg("World", id="m2")

    # Pre-populate DB with m1 before run_graph (use message ID "m1" matching the pickle)
    pre_db = GraphDB(db_path)
    pre_db.upsert_node("m1", label="Message")
    pre_db.upsert_node("c1", label="Conversation")
    pre_db.add_edge_contains("c1", "m1")
    pre_db.close()

    msgs = [msg1, msg2]
    msgs_pkl = tmp_path / "messages.pkl"
    with open(msgs_pkl, "wb") as f:
        pickle.dump(msgs, f)

    def mock_nlp_side(text: str):
        return _make_mock_doc([])

    mock_nlp = MagicMock()
    mock_nlp.side_effect = mock_nlp_side

    with patch("spacy.load", return_value=mock_nlp):
        with patch("sklearn.feature_extraction.text.TfidfVectorizer") as mock_tfidf:
            mock_v = MagicMock()
            mock_tfidf.return_value = mock_v
            mock_v.fit_transform.return_value = MagicMock()
            run_graph(pickle_path=msgs_pkl, db_path=db_path, export_pickle=True)

    loaded = pickle.loads((tmp_path / "knowledge_graph.pkl").read_bytes())
    assert "m2" in loaded["nodes"]
    assert "m1" in loaded["nodes"]


def test_run_graph_up_to_date(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "knowledge_graph.db"
    msg = _msg("Hello", id="m1")
    msgs_pkl = tmp_path / "messages.pkl"
    with open(msgs_pkl, "wb") as f:
        pickle.dump([msg], f)

    # Pre-populate DB with m1 so run_graph sees it as up to date
    pre_db = GraphDB(db_path)
    pre_db.upsert_node("m1", label="Message")
    pre_db.upsert_node("c1", label="Conversation")
    pre_db.add_edge_contains("c1", "m1")
    pre_db.mark_messages_processed({"m1"})
    pre_db.close()

    with patch("spacy.load"):
        with patch("sklearn.feature_extraction.text.TfidfVectorizer"):
            run_graph(pickle_path=msgs_pkl, db_path=db_path, export_pickle=False)

    assert "up to date" in capsys.readouterr().out


def test_run_graph_offset_limit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    msgs = [_msg(f"Msg {i}", id=f"m{i}") for i in range(5)]
    msgs_pkl = tmp_path / "messages.pkl"
    with open(msgs_pkl, "wb") as f:
        pickle.dump(msgs, f)

    def mock_nlp_side(text: str):
        return _make_mock_doc([])

    mock_nlp = MagicMock()
    mock_nlp.side_effect = mock_nlp_side

    with patch("spacy.load", return_value=mock_nlp):
        with patch("sklearn.feature_extraction.text.TfidfVectorizer") as mock_tfidf:
            mock_v = MagicMock()
            mock_tfidf.return_value = mock_v
            mock_v.fit_transform.return_value = MagicMock()
            run_graph(pickle_path=msgs_pkl, db_path=tmp_path / "knowledge_graph.db", offset=2, limit=2, export_pickle=True)

    loaded = pickle.loads((tmp_path / "knowledge_graph.pkl").read_bytes())
    msg_nodes = [k for k in loaded["nodes"] if k.startswith("m")]
    assert len(msg_nodes) == 2
    assert "m2" in loaded["nodes"]
    assert "m3" in loaded["nodes"]


def test_run_graph_malformed_existing(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    db_path = tmp_path / "knowledge_graph.db"
    msg = _msg("Hello", id="m1")
    msgs_pkl = tmp_path / "messages.pkl"
    with open(msgs_pkl, "wb") as f:
        pickle.dump([msg], f)

    with patch("spacy.load"):
        with patch("sklearn.feature_extraction.text.TfidfVectorizer"):
            run_graph(pickle_path=msgs_pkl, db_path=db_path)

    out_lower = capsys.readouterr().out.lower()
    assert "loaded" in out_lower


def test_run_graph_empty_messages(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    msgs_pkl = tmp_path / "messages.pkl"
    with open(msgs_pkl, "wb") as f:
        pickle.dump([], f)

    with patch("spacy.load"):
        with patch("sklearn.feature_extraction.text.TfidfVectorizer"):
            run_graph(pickle_path=msgs_pkl, db_path=tmp_path / "knowledge_graph.db")

    out = capsys.readouterr().out
    assert "No new messages" in out or "Done" in out
