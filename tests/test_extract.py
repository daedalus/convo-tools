from __future__ import annotations

from convo_tools import extract_messages


def test_extract_messages_normal(sample_conversation: dict) -> None:
    messages = extract_messages(sample_conversation)
    assert len(messages) == 2
    assert messages[0]["id"] == "msg1"
    assert messages[0]["role"] == "user"
    assert messages[0]["text"] == "Hello, world!"
    assert messages[0]["parent"] is None
    assert messages[1]["id"] == "msg2"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["text"] == "Hi there!\nHow can I help?"
    assert messages[1]["parent"] == "msg1"


def test_extract_messages_non_text_skipped() -> None:
    conversation = {
        "mapping": {
            "m1": {
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["hello"]},
                },
            },
            "m2": {
                "message": {
                    "author": {"role": "assistant"},
                    "content": {"content_type": "code", "parts": ["print(1)"]},
                },
            },
        },
    }
    messages = extract_messages(conversation)
    assert len(messages) == 1
    assert messages[0]["id"] == "m1"


def test_extract_messages_empty_mapping() -> None:
    conversation = {"mapping": {}}
    assert extract_messages(conversation) == []


def test_extract_messages_missing_message_field() -> None:
    conversation = {
        "mapping": {
            "m1": {
                "message": None,
            },
            "m2": {},
        },
    }
    assert extract_messages(conversation) == []


def test_extract_messages_missing_author_role() -> None:
    conversation = {
        "mapping": {
            "m1": {
                "message": {
                    "author": {},
                    "content": {"content_type": "text", "parts": ["test"]},
                },
            },
        },
    }
    messages = extract_messages(conversation)
    assert len(messages) == 1
    assert messages[0]["role"] == "unknown"


def test_extract_messages_conversation_id_from_arg() -> None:
    conversation = {
        "mapping": {
            "m1": {
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["test"]},
                },
            },
        },
    }
    messages = extract_messages(conversation)
    assert len(messages) == 1
    assert "conversation_id" not in messages[0]
