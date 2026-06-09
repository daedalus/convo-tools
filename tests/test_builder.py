from __future__ import annotations

from unittest.mock import patch

from convo_tools._builder import build_graph
from convo_tools._util import text_hash


def test_text_hash_consistency() -> None:
    h1 = text_hash("hello")
    h2 = text_hash("hello")
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hexdigest


def test_text_hash_different() -> None:
    assert text_hash("hello") != text_hash("world")


def test_text_hash_empty() -> None:
    h = text_hash("")
    assert len(h) == 64


class TestBuildGraph:
    """Tests for build_graph()."""

    @patch("spacy.load")
    def test_empty_messages(self, mock_load) -> None:
        mock_nlp = mock_load.return_value
        mock_nlp.max_length = 100_000

        graph = build_graph([])
        assert graph["nodes"] == {}
        assert graph["edges_contains"] == set()
        assert graph["edges_replies_to"] == set()
        assert graph["edges_mentions"] == set()
        assert graph["edges_cooc"] == set()
        assert graph["edges_keywords"] == []

    @patch("spacy.load")
    def test_single_conversation(self, mock_load, sample_messages: list[dict]) -> None:
        mock_nlp = mock_load.return_value
        mock_nlp.max_length = 100_000

        graph = build_graph(sample_messages)
        assert "conv1" in graph["nodes"]
        assert graph["nodes"]["conv1"] == {"label": "Conversation"}
        assert "msg1" in graph["nodes"]
        assert graph["nodes"]["msg1"]["role"] == "user"
        assert ("conv1", "msg1") in graph["edges_contains"]
        assert ("msg1", "msg2") in graph["edges_replies_to"]
        assert ("msg2", "msg3") in graph["edges_replies_to"]

    @patch("spacy.load")
    def test_orphan_parent_skipped(self, mock_load) -> None:
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
        graph = build_graph(msgs)
        assert len(graph["edges_replies_to"]) == 0

    @patch("spacy.load")
    def test_limit(self, mock_load) -> None:
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
        graph = build_graph(msgs, limit=3)
        msg_ids = {k for k, v in graph["nodes"].items() if v.get("label") == "Message"}
        assert len(msg_ids) == 3
        assert ("c1", "msg0") in graph["edges_contains"]
        assert ("c1", "msg2") in graph["edges_contains"]
        assert ("c1", "msg9") not in graph["edges_contains"]
