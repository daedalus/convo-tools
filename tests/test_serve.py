from __future__ import annotations

import pytest

from convo_tools import _serve


def _graph() -> dict:
    return {
        "nodes": {
            "conv::c1": {"label": "Conversation"},
            "msg::1": {"label": "Message", "role": "user", "text": "Hello world about AI"},
            "msg::2": {"label": "Message", "role": "assistant", "text": "AI is interesting"},
            "msg::3": {"label": "Message", "role": "user", "text": "Tell me more"},
            "msg::4": {"label": "Message", "role": "assistant", "text": "Alice works at OpenAI"},
            "entity::PERSON::alice": {"label": "Entity", "name": "alice", "entity_type": "PERSON"},
            "entity::ORG::openai": {"label": "Entity", "name": "openai", "entity_type": "ORG"},
            "entity::TOPIC::ai": {"label": "Entity", "name": "ai", "entity_type": "TOPIC"},
            "kw::hello": {"label": "Keyword", "name": "hello"},
            "kw::interesting": {"label": "Keyword", "name": "interesting"},
        },
        "edges_contains": {
            ("conv::c1", "msg::1"),
            ("conv::c1", "msg::2"),
            ("conv::c1", "msg::3"),
            ("conv::c1", "msg::4"),
        },
        "edges_replies_to": {("msg::1", "msg::2"), ("msg::2", "msg::3")},
        "edges_mentions": {
            ("msg::1", "entity::PERSON::alice"),
            ("msg::2", "entity::TOPIC::ai"),
            ("msg::4", "entity::PERSON::alice"),
            ("msg::4", "entity::ORG::openai"),
        },
        "edges_cooc": {
            ("entity::PERSON::alice", "entity::ORG::openai"),
            ("entity::PERSON::alice", "entity::TOPIC::ai"),
            ("entity::ORG::openai", "entity::TOPIC::ai"),
        },
        "edges_keywords": [
            ("msg::1", "kw::hello", 0.5),
            ("msg::2", "kw::interesting", 0.8),
        ],
    }


def _messages() -> list[dict]:
    return [
        {"id": "msg::1", "conversation_id": "conv::c1", "role": "user", "text": "Hello world about AI", "create_time": 1000000.0},
        {"id": "msg::2", "conversation_id": "conv::c1", "role": "assistant", "text": "AI is interesting", "create_time": 1000100.0},
        {"id": "msg::3", "conversation_id": "conv::c1", "role": "user", "text": "Tell me more", "create_time": 1000200.0},
        {"id": "msg::4", "conversation_id": "conv::c1", "role": "assistant", "text": "Alice works at OpenAI", "create_time": 1000300.0},
    ]


@pytest.fixture(autouse=True)
def _mock_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_serve, "_g", _graph)
    monkeypatch.setattr(_serve, "_m", _messages)


# ── Basic tools ──


class TestGraphStats:
    def test_returns_counts(self) -> None:
        r = _serve.graph_stats()
        assert r["nodes"]["total"] == 10
        assert r["nodes"]["by_label"]["Entity"] == 3
        assert r["nodes"]["by_label"]["Message"] == 4
        assert r["edges"]["MENTIONS"] == 4
        assert r["edges"]["CO_OCCURS_WITH"] == 3


class TestSearchEntities:
    def test_substring_match(self) -> None:
        r = _serve.search_entities("ali")
        assert len(r) == 1
        assert r[0]["name"] == "alice"

    def test_entity_type_filter(self) -> None:
        r = _serve.search_entities("a", entity_type="ORG")
        assert len(r) == 1
        assert r[0]["name"] == "openai"

    def test_no_match(self) -> None:
        assert _serve.search_entities("zzzz") == []

    def test_limit(self) -> None:
        r = _serve.search_entities("a", limit=1)
        assert len(r) == 1


class TestSearchKeywords:
    def test_substring(self) -> None:
        r = _serve.search_keywords("ello")
        assert len(r) == 1
        assert r[0]["name"] == "hello"

    def test_no_match(self) -> None:
        assert _serve.search_keywords("zzzz") == []

    def test_avg_tfidf(self) -> None:
        r = _serve.search_keywords("hello")
        assert r[0]["avg_tfidf"] == 0.5


class TestGetEntityMessages:
    def test_messages_found(self) -> None:
        r = _serve.get_entity_messages("entity::PERSON::alice")
        assert len(r) == 2
        assert any(m["role"] == "user" for m in r)
        assert any(m["role"] == "assistant" for m in r)

    def test_limit(self) -> None:
        r = _serve.get_entity_messages("entity::PERSON::alice", limit=1)
        assert len(r) == 1

    def test_unknown_entity(self) -> None:
        r = _serve.get_entity_messages("entity::nonexistent")
        assert "error" in r[0]


class TestGetKeywordMessages:
    def test_messages_found(self) -> None:
        r = _serve.get_keyword_messages("kw::hello")
        assert len(r) == 1
        assert r[0]["tfidf_score"] == 0.5

    def test_min_tfidf_filter(self) -> None:
        r = _serve.get_keyword_messages("kw::hello", min_tfidf=0.6)
        assert len(r) == 0

    def test_unknown_keyword(self) -> None:
        r = _serve.get_keyword_messages("kw::nonexistent")
        assert "error" in r[0]


class TestGetConversation:
    def test_full(self) -> None:
        r = _serve.get_conversation("conv::c1")
        assert r["message_count"] == 4
        assert len(r["messages"]) == 4

    def test_without_messages(self) -> None:
        r = _serve.get_conversation("conv::c1", include_messages=False)
        assert "messages" not in r

    def test_unknown(self) -> None:
        r = _serve.get_conversation("nonexistent")
        assert "error" in r


class TestGetMessageContext:
    def test_ancestors_and_descendants(self) -> None:
        r = _serve.get_message_context("msg::2")
        assert len(r["ancestors"]) == 1
        assert r["ancestors"][0]["id"] == "msg::1"
        assert len(r["descendants"]) == 1
        assert r["descendants"][0]["id"] == "msg::3"

    def test_unknown(self) -> None:
        r = _serve.get_message_context("msg::nonexistent")
        assert "error" in r


class TestCoOccurringEntities:
    def test_peers_found(self) -> None:
        r = _serve.co_occurring_entities("entity::PERSON::alice")
        assert len(r) == 2
        names = {e["name"] for e in r}
        assert "openai" in names
        assert "ai" in names

    def test_unknown(self) -> None:
        r = _serve.co_occurring_entities("entity::nonexistent")
        assert "error" in r[0]


class TestTopEntities:
    def test_all_types(self) -> None:
        r = _serve.top_entities()
        assert len(r) == 3
        assert r[0]["name"] == "alice"
        assert r[0]["mention_count"] == 2

    def test_type_filter(self) -> None:
        r = _serve.top_entities(entity_type="ORG")
        assert len(r) == 1
        assert r[0]["name"] == "openai"

    def test_limit(self) -> None:
        r = _serve.top_entities(limit=1)
        assert len(r) == 1


class TestTopKeywords:
    def test_ranking(self) -> None:
        r = _serve.top_keywords()
        assert len(r) == 2

    def test_limit(self) -> None:
        r = _serve.top_keywords(limit=1)
        assert len(r) == 1


class TestSearchMessageText:
    def test_found(self) -> None:
        r = _serve.search_message_text("AI")
        assert len(r) >= 2

    def test_role_filter(self) -> None:
        r = _serve.search_message_text("AI", role="user")
        assert len(r) == 1
        assert r[0]["role"] == "user"

    def test_not_found(self) -> None:
        r = _serve.search_message_text("xyznonexistent")
        assert r == []


class TestEntityTimeline:
    def test_single(self) -> None:
        r = _serve.entity_timeline("entity::PERSON::alice")
        assert r["entity"]["name"] == "alice"
        assert r["conversation_count"] == 1
        assert "conv::c1" in r["conversation_ids"]

    def test_pair(self) -> None:
        r = _serve.entity_timeline("entity::PERSON::alice", "entity::ORG::openai")
        assert r["co_mention_count"] == 1

    def test_unknown(self) -> None:
        r = _serve.entity_timeline("entity::nonexistent")
        assert "error" in r


# ── Advanced tools ──


class TestEntityCentrality:
    def test_ranking(self) -> None:
        r = _serve.entity_centrality()
        assert len(r) == 3
        for item in r:
            assert "betweenness" in item
            assert "degree" in item

    def test_type_filter(self) -> None:
        r = _serve.entity_centrality(entity_type="PERSON")
        assert len(r) == 1

    def test_empty_graph(self) -> None:
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(_serve, "_g", lambda: {"nodes": {}, "edges_cooc": set(), "edges_contains": set(), "edges_replies_to": set(), "edges_mentions": set(), "edges_keywords": []})
        result = _serve.entity_centrality()
        assert result == []
        monkeypatch.undo()


class TestSimilarConversations:
    def test_similar_found(self, monkeypatch) -> None:
        g2 = dict(_graph())
        g2["nodes"]["conv::c2"] = {"label": "Conversation"}
        g2["nodes"]["msg::5"] = {"label": "Message", "role": "user", "text": "Hello world about AI too"}
        g2["edges_contains"] = g2["edges_contains"] | {("conv::c2", "msg::5")}
        g2["edges_mentions"] = g2["edges_mentions"] | {("msg::5", "entity::PERSON::alice")}
        monkeypatch.setattr(_serve, "_g", lambda: g2)
        r = _serve.similar_conversations("conv::c1", threshold=0.0)
        assert len(r) >= 1

    def test_unknown(self) -> None:
        r = _serve.similar_conversations("conv::nonexistent")
        assert "error" in r[0]


class TestTopicClusters:
    def test_clusters_found(self) -> None:
        r = _serve.topic_clusters(min_size=2)
        assert len(r) >= 1
        assert "top_entities" in r[0]
        assert "top_keywords" in r[0]

    def test_types(self) -> None:
        r = _serve.topic_clusters(min_size=2)
        for cluster in r:
            for ent in cluster["top_entities"]:
                assert "name" in ent
                assert "degree" in ent


class TestReplyChainStats:
    def test_chain_found(self) -> None:
        r = _serve.reply_chain_stats("conv::c1")
        assert r["max_depth"] >= 2
        assert r["message_count_in_chain"] >= 3

    def test_unknown(self) -> None:
        r = _serve.reply_chain_stats("conv::nonexistent")
        assert "error" in r


class TestEntityTemporalMetrics:
    def test_metrics(self) -> None:
        r = _serve.entity_temporal_metrics("entity::PERSON::alice")
        assert r["entity"]["name"] == "alice"
        assert r["total_mentions"] == 2
        assert r["active_buckets"] >= 1

    def test_unknown(self) -> None:
        r = _serve.entity_temporal_metrics("entity::nonexistent")
        assert "error" in r


class TestEntityTimelineBucket:
    def test_specific_bucket(self) -> None:
        r = _serve.entity_timeline_bucket("", freq="month", top=5)
        assert len(r) >= 3

    def test_bucket_not_found(self) -> None:
        r = _serve.entity_timeline_bucket("2999-99")
        assert "error" in r[0]
