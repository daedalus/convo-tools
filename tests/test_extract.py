from __future__ import annotations

from convo_tools import extract_messages
from convo_tools._extract import _is_base64, _is_noise


def test_is_base64_valid() -> None:
    import base64
    payload = base64.b64encode(b"hello world this is a test message with enough length to pass the 100 char check").decode()
    assert _is_base64(payload) is True


def test_is_base64_data_uri() -> None:
    assert _is_base64("data:image/png;base64,iVBORw0KGgo=") is True


def test_is_base64_short_string() -> None:
    assert _is_base64("hello") is False


def test_is_base64_normal_text() -> None:
    assert _is_base64("This is normal English text with spaces and punctuation. It has enough characters but is clearly not base64 encoded data at all.") is False


def test_is_base64_invalid_chars() -> None:
    long_invalid = "a" * 100 + "!!!"
    assert _is_base64(long_invalid) is False


def test_is_base64_invalid_padding() -> None:
    invalid_pad = "A" * 100 + "==="
    assert _is_base64(invalid_pad) is False


def test_is_base64_empty() -> None:
    assert _is_base64("") is False
    assert _is_base64("   ") is False


def test_extract_messages_filters_base64() -> None:
    import base64
    blob = base64.b64encode(b"x" * 200).decode()
    conversation = {
        "mapping": {
            "m1": {
                "message": {
                    "author": {"role": "user"},
                    "content": {"content_type": "text", "parts": ["Hello", blob, "World"]},
                },
            },
        },
    }
    messages = extract_messages(conversation)
    assert len(messages) == 1
    assert messages[0]["text"] == "Hello\nWorld"


def test_is_noise_image_asset_pointer() -> None:
    assert _is_noise("{'content_type': 'image_asset_pointer', 'asset_pointer': 'data:image/png;base64,abc'}") is True


def test_is_noise_markdown_image() -> None:
    assert _is_noise("![image](data:image/png;base64,iVBORw0KGgo)") is True


def test_is_noise_normal_text() -> None:
    assert _is_noise("This is normal text about Python performance.") is False


def test_is_noise_short_asset_pointer() -> None:
    assert _is_noise("image_asset_pointer") is True


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
