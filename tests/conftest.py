from __future__ import annotations

from unittest.mock import MagicMock

import pytest


@pytest.fixture
def sample_conversation() -> dict:
    return {
        "mapping": {
            "msg1": {
                "message": {
                    "author": {"role": "user"},
                    "content": {
                        "content_type": "text",
                        "parts": ["Hello, world!"],
                    },
                },
                "parent": None,
            },
            "msg2": {
                "message": {
                    "author": {"role": "assistant"},
                    "content": {
                        "content_type": "text",
                        "parts": ["Hi there!", "How can I help?"],
                    },
                },
                "parent": "msg1",
            },
        },
    }


@pytest.fixture
def sample_messages() -> list[dict]:
    return [
        {
            "id": "msg1",
            "role": "user",
            "text": "Hello world",
            "parent": None,
            "conversation_id": "conv1",
        },
        {
            "id": "msg2",
            "role": "assistant",
            "text": "Hi there!",
            "parent": "msg1",
            "conversation_id": "conv1",
        },
        {
            "id": "msg3",
            "role": "user",
            "text": "Tell me about Fibonacci numbers",
            "parent": "msg2",
            "conversation_id": "conv1",
        },
    ]


@pytest.fixture
def mock_spacy_nlp():
    """Create a mock spaCy nlp object that returns controlled entities."""
    mock_nlp = MagicMock()
    mock_nlp.max_length = 100_000

    def _make_doc(text):
        doc = MagicMock()
        doc.ents = []
        if "Fibonacci" in text:
            ent = MagicMock()
            ent.label_ = "PERSON"
            ent.text = "Fibonacci"
            doc.ents = [ent]
        return doc

    mock_nlp.side_effect = _make_doc
    return mock_nlp



