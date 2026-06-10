from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from convo_tools._query import run_query


def _graph() -> dict:
    return {
        "nodes": {
            "conv::c1": {"label": "Conversation"},
            "msg::1": {"label": "Message", "role": "user", "text": "Tell me about Fibonacci numbers"},
            "msg::2": {"label": "Message", "role": "assistant", "text": "Fibonacci is a sequence"},
            "entity::TOPIC::fibonacci": {"label": "Entity", "name": "fibonacci", "entity_type": "TOPIC"},
            "entity::TOPIC::numbers": {"label": "Entity", "name": "numbers", "entity_type": "TOPIC"},
            "kw::sequence": {"label": "Keyword", "name": "sequence"},
        },
        "edges_mentions": {
            ("msg::1", "entity::TOPIC::fibonacci"),
            ("msg::2", "entity::TOPIC::fibonacci"),
            ("msg::1", "entity::TOPIC::numbers"),
        },
        "edges_contains": {("conv::c1", "msg::1"), ("conv::c1", "msg::2")},
        "edges_replies_to": {("msg::1", "msg::2")},
        "edges_cooc": {("entity::TOPIC::fibonacci", "entity::TOPIC::numbers")},
        "edges_keywords": [("msg::2", "kw::sequence", 0.6)],
    }


def _messages() -> list[dict]:
    return [
        {"id": "msg::1", "conversation_id": "conv::c1", "role": "user", "text": "Tell me about Fibonacci numbers"},
        {"id": "msg::2", "conversation_id": "conv::c1", "role": "assistant", "text": "Fibonacci is a sequence"},
    ]


def test_query_keyword_mode(tmp_path: Path, capsys) -> None:
    pkl_g = tmp_path / "graph.pkl"
    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump(_graph(), f)
    with open(pkl_m, "wb") as f:
        pickle.dump(_messages(), f)

    args = argparse.Namespace(query="fibonacci", graph=pkl_g, messages=pkl_m, llm=False, top=10, max_context=50000, output=None)
    run_query(args)
    out = capsys.readouterr().out
    assert "fibonacci" in out.lower()


def test_query_text_fallback(tmp_path: Path, capsys) -> None:
    g = {
        "nodes": {"conv::c1": {"label": "Conversation"}, "msg::1": {"label": "Message", "role": "user", "text": "Hello world"}},
        "edges_mentions": set(),
        "edges_contains": {("conv::c1", "msg::1")},
        "edges_replies_to": set(), "edges_cooc": set(), "edges_keywords": [],
    }
    pkl_g = tmp_path / "graph.pkl"
    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump(g, f)
    with open(pkl_m, "wb") as f:
        pickle.dump([
            {"id": "msg::1", "conversation_id": "conv::c1", "role": "user", "text": "Alice likes apples and bananas", "create_time": 1000.0},
            {"id": "msg::2", "conversation_id": "conv::c1", "role": "assistant", "text": "apples are red fruits", "create_time": 1001.0},
            {"id": "msg::3", "conversation_id": "conv::c1", "role": "user", "text": "I eat bananas every day fruits", "create_time": 1002.0},
            {"id": "msg::4", "conversation_id": "conv::c1", "role": "assistant", "text": "apples and bananas are both fruits", "create_time": 1003.0},
            {"id": "msg::5", "conversation_id": "conv::c1", "role": "user", "text": "Bananas are yellow fruits", "create_time": 1004.0},
        ], f)

    args = argparse.Namespace(query="apples", graph=pkl_g, messages=pkl_m, llm=False, top=10, max_context=50000, output=None)
    run_query(args)
    out = capsys.readouterr().out
    assert "apple" in out or "Alice" in out


def test_query_csv(tmp_path: Path) -> None:
    pkl_g = tmp_path / "graph.pkl"
    pkl_m = tmp_path / "messages.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump(_graph(), f)
    with open(pkl_m, "wb") as f:
        pickle.dump(_messages(), f)
    out_csv = tmp_path / "out.csv"

    args = argparse.Namespace(query="fibonacci", graph=pkl_g, messages=pkl_m, llm=False, top=10, max_context=50000, output=out_csv)
    run_query(args)
    assert out_csv.exists()


def test_query_no_messages_pickle(tmp_path: Path, capsys) -> None:
    pkl_g = tmp_path / "graph.pkl"
    with open(pkl_g, "wb") as f:
        pickle.dump(_graph(), f)

    args = argparse.Namespace(query="fibonacci", graph=pkl_g, messages=Path("/nonexistent/pkl"), llm=False, top=10, max_context=50000, output=None)
    run_query(args)
    out = capsys.readouterr().out
    assert "fibonacci" in out.lower() or "matching" in out
