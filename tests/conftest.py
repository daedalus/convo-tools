from __future__ import annotations

import pytest
from convo_tools._builder import _LANG_MODELS


@pytest.fixture(autouse=True)
def _clear_lang_model_cache() -> None:
    _LANG_MODELS.clear()


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
            "lang": "en",
        },
        {
            "id": "msg2",
            "role": "assistant",
            "text": "Hi there!",
            "parent": "msg1",
            "conversation_id": "conv1",
            "lang": "en",
        },
        {
            "id": "msg3",
            "role": "user",
            "text": "Tell me about Fibonacci numbers",
            "parent": "msg2",
            "conversation_id": "conv1",
            "lang": "en",
        },
    ]


