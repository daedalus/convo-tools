from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from convo_tools._builder import build_graph_to_db
from convo_tools._graph_db import GraphDB
from convo_tools._util import text_hash


def test_text_hash_consistency() -> None:
    h1 = text_hash("hello")
    h2 = text_hash("hello")
    assert h1 == h2
    assert len(h1) == 64


def test_text_hash_different() -> None:
    assert text_hash("hello") != text_hash("world")


def test_text_hash_empty() -> None:
    h = text_hash("")
    assert len(h) == 64


class TestBuildGraph:
    """Tests for build_graph_to_db()."""

    @patch("spacy.load")
    def test_empty_messages(self, mock_load, tmp_path: Path) -> None:
        mock_nlp = mock_load.return_value
        mock_nlp.max_length = 100_000

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

    @patch("spacy.load")
    def test_single_conversation(self, mock_load, tmp_path: Path, sample_messages: list[dict]) -> None:
        mock_nlp = mock_load.return_value
        mock_nlp.max_length = 100_000

        db = GraphDB(tmp_path / "test.db")
        build_graph_to_db(sample_messages, db)
        assert db.get_node("conv1") is not None
        assert db.get_node("conv1")["label"] == "Conversation"
        assert db.get_node("msg1") is not None
        assert db.get_node("msg1")["role"] == "user"
        edges_contains = set(db.get_edges_contains())
        assert ("conv1", "msg1") in edges_contains
        edges_replies = set(db.get_edges_replies_to())
        assert ("msg1", "msg2") in edges_replies
        assert ("msg2", "msg3") in edges_replies
        db.close()

    @patch("spacy.load")
    def test_orphan_parent_skipped(self, mock_load, tmp_path: Path) -> None:
        mock_nlp = mock_load.return_value
        mock_nlp.max_length = 100_000

        msgs = [
            {
                "id": "msg2",
                "role": "assistant",
                "text": "reply",
                "parent": "nonexistent",
                "conversation_id": "c1",
            },
        ]
        db = GraphDB(tmp_path / "test.db")
        build_graph_to_db(msgs, db)
        assert db.get_edges_replies_to() == []
        db.close()

    @patch("spacy.load")
    def test_limit(self, mock_load, tmp_path: Path) -> None:
        mock_nlp = mock_load.return_value
        mock_nlp.max_length = 100_000

        msgs = [
            {
                "id": f"msg{i}",
                "role": "user",
                "text": f"text{i}",
                "parent": None,
                "conversation_id": "c1",
            }
            for i in range(10)
        ]
        db = GraphDB(tmp_path / "test.db")
        build_graph_to_db(msgs[:3], db)
        msg_nodes = db.get_all_nodes_by_label("Message")
        assert len(msg_nodes) == 3
        msg_ids = {n["id"] for n in msg_nodes}
        assert "msg0" in msg_ids
        assert "msg2" in msg_ids
        assert "msg9" not in msg_ids
        edges_contains = set(db.get_edges_contains())
        assert ("c1", "msg0") in edges_contains
        assert ("c1", "msg2") in edges_contains
        assert ("c1", "msg9") not in edges_contains
        db.close()
