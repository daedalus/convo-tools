from __future__ import annotations

import math
import pickle
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP: Any = None  # type: ignore[no-redef]


P = Path.home() / ".convo-tools"

_graph: dict[str, Any] | None = None
_GRAPH_PATH: Path = P / "knowledge_graph.pkl"
_messages: list[dict[str, Any]] | None = None
_MESSAGES_PATH: Path = P / "messages.pkl"


def _load_graph() -> dict[str, Any]:
    global _graph
    if _graph is None:
        if not _GRAPH_PATH.exists():
            raise FileNotFoundError(
                f"Graph pickle not found: {_GRAPH_PATH}. "
                "Run: convo-tools -m graph messages.pkl --pickle"
            )
        with open(_GRAPH_PATH, "rb") as f:
            _graph = pickle.load(f)
    return _graph


def _g() -> dict[str, Any]:
    return _load_graph()


def _load_messages() -> list[dict[str, Any]]:
    global _messages
    if _messages is None:
        if not _MESSAGES_PATH.exists():
            return []
        with open(_MESSAGES_PATH, "rb") as f:
            _messages = pickle.load(f)
    return _messages


def _m() -> list[dict[str, Any]]:
    return _load_messages()


mcp = FastMCP(
    "convo-graph",
    instructions=(
        "Query a knowledge graph built from LLM conversation exports. "
        "Nodes: Conversation, Message, Entity, Keyword. "
        "Edges: CONTAINS, REPLIES_TO, MENTIONS, CO_OCCURS_WITH, HAS_KEYWORD. "
        "Use search_entities or search_keywords first to discover node IDs, "
        "then use the traversal tools to explore context."
    ),
)


@mcp.tool()
def graph_stats() -> dict[str, Any]:
    """Return high-level statistics about the knowledge graph."""
    g = _g()
    nodes = g["nodes"]

    label_counts: Counter[str] = Counter()
    entity_type_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    for node_id, attrs in nodes.items():
        label = attrs.get("label", "unknown")
        label_counts[label] += 1
        if label == "Entity":
            entity_type_counts[attrs.get("entity_type", "?")] += 1
        if label == "Message":
            role_counts[attrs.get("role", "?")] += 1

    return {
        "nodes": {
            "total": len(nodes),
            "by_label": dict(label_counts),
            "entity_types": dict(entity_type_counts.most_common(20)),
            "message_roles": dict(role_counts),
        },
        "edges": {
            "CONTAINS": len(g["edges_contains"]),
            "REPLIES_TO": len(g["edges_replies_to"]),
            "MENTIONS": len(g["edges_mentions"]),
            "CO_OCCURS_WITH": len(g["edges_cooc"]),
            "HAS_KEYWORD": len(g["edges_keywords"]),
        },
    }


@mcp.tool()
def search_entities(
    query: str,
    entity_type: str = "",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Search entity nodes by name substring.

    Args:
        query: Substring to match against entity name (case-insensitive).
        entity_type: Optional spaCy NER label filter (e.g. PERSON, ORG, GPE,
                     PRODUCT, DATE, NORP, LOC, WORK_OF_ART, EVENT, LANGUAGE).
                     Leave empty to search all types.
        limit: Maximum results to return (default 20).

    Returns:
        List of matching entity dicts with id, name, entity_type, mention_count.
    """
    g = _g()
    nodes = g["nodes"]
    edges_mentions: set[tuple[str, str]] = g["edges_mentions"]

    mention_counts: Counter[str] = Counter(eid for _, eid in edges_mentions)

    q = query.lower()
    results = []
    for node_id, attrs in nodes.items():
        if attrs.get("label") != "Entity":
            continue
        name = attrs.get("name", "")
        etype = attrs.get("entity_type", "")
        if q not in name:
            continue
        if entity_type and etype != entity_type.upper():
            continue
        results.append({
            "id": node_id,
            "name": name,
            "entity_type": etype,
            "mention_count": mention_counts.get(node_id, 0),
        })

    results.sort(key=lambda x: -x["mention_count"])
    return results[:limit]


@mcp.tool()
def search_keywords(
    query: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Search keyword nodes by name substring.

    Args:
        query: Substring to match against keyword (case-insensitive).
        limit: Maximum results to return (default 20).

    Returns:
        List of matching keyword dicts with id, name, message_count, avg_tfidf.
    """
    g = _g()
    nodes = g["nodes"]
    edges_keywords: list[tuple[str, str, float]] = g["edges_keywords"]

    kw_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "total_score": 0.0}
    )
    for _msg_id, kw_id, score in edges_keywords:
        kw_stats[kw_id]["count"] += 1
        kw_stats[kw_id]["total_score"] += score

    q = query.lower()
    results = []
    for node_id, attrs in nodes.items():
        if attrs.get("label") != "Keyword":
            continue
        name = attrs.get("name", "")
        if q not in name:
            continue
        stats = kw_stats.get(node_id, {"count": 0, "total_score": 0.0})
        count = stats["count"]
        avg = stats["total_score"] / count if count else 0.0
        results.append({
            "id": node_id,
            "name": name,
            "message_count": count,
            "avg_tfidf": round(avg, 4),
        })

    results.sort(key=lambda x: (-x["message_count"], -x["avg_tfidf"]))
    return results[:limit]


@mcp.tool()
def get_entity_messages(
    entity_id: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Get messages that mention a given entity.

    Args:
        entity_id: Entity node ID (from search_entities).
        limit: Maximum messages to return (default 10).

    Returns:
        List of message dicts with id, role, text_preview, conversation_id.
    """
    g = _g()
    nodes = g["nodes"]
    edges_mentions: set[tuple[str, str]] = g["edges_mentions"]
    edges_contains: set[tuple[str, str]] = g["edges_contains"]

    if entity_id not in nodes:
        return [{"error": f"entity not found: {entity_id}"}]

    msg_to_conv: dict[str, str] = {mid: cid for cid, mid in edges_contains}

    results = []
    for msg_id, eid in edges_mentions:
        if eid != entity_id:
            continue
        msg = nodes.get(msg_id, {})
        results.append({
            "id": msg_id,
            "role": msg.get("role", "?"),
            "text_preview": msg.get("text", "")[:300],
            "conversation_id": msg_to_conv.get(msg_id, ""),
        })
        if len(results) >= limit:
            break

    return results


@mcp.tool()
def get_keyword_messages(
    keyword_id: str,
    limit: int = 10,
    min_tfidf: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Get messages associated with a keyword, sorted by TF-IDF score.

    Args:
        keyword_id: Keyword node ID (from search_keywords).
        limit: Maximum messages to return (default 10).
        min_tfidf: Minimum TF-IDF score threshold (default 0.0).

    Returns:
        List of message dicts with id, role, text_preview, tfidf_score.
    """
    g = _g()
    nodes = g["nodes"]
    edges_keywords: list[tuple[str, str, float]] = g["edges_keywords"]
    edges_contains: set[tuple[str, str]] = g["edges_contains"]

    if keyword_id not in nodes:
        return [{"error": f"keyword not found: {keyword_id}"}]

    msg_to_conv: dict[str, str] = {mid: cid for cid, mid in edges_contains}

    hits = [
        (msg_id, score)
        for msg_id, kw_id, score in edges_keywords
        if kw_id == keyword_id and score >= min_tfidf
    ]
    hits.sort(key=lambda x: -x[1])

    results = []
    for msg_id, score in hits[:limit]:
        msg = nodes.get(msg_id, {})
        results.append({
            "id": msg_id,
            "role": msg.get("role", "?"),
            "text_preview": msg.get("text", "")[:300],
            "tfidf_score": round(score, 4),
            "conversation_id": msg_to_conv.get(msg_id, ""),
        })

    return results


@mcp.tool()
def get_conversation(
    conversation_id: str,
    include_messages: bool = True,
) -> dict[str, Any]:
    """
    Get a conversation node and optionally all its messages in reply order.

    Args:
        conversation_id: Conversation node ID.
        include_messages: Whether to include full message list (default True).

    Returns:
        Dict with conversation metadata and messages list.
    """
    g = _g()
    nodes = g["nodes"]
    edges_contains: set[tuple[str, str]] = g["edges_contains"]
    edges_replies_to: set[tuple[str, str]] = g["edges_replies_to"]

    if conversation_id not in nodes:
        return {"error": f"conversation not found: {conversation_id}"}

    if not include_messages:
        return {"id": conversation_id, **nodes[conversation_id]}

    msg_ids = {mid for cid, mid in edges_contains if cid == conversation_id}

    children: dict[str, list[str]] = defaultdict(list)
    for parent_id, child_id in edges_replies_to:
        if child_id in msg_ids:
            children[parent_id].append(child_id)

    has_parent = {child for _, child in edges_replies_to if child in msg_ids}
    roots = msg_ids - has_parent

    ordered: list[str] = []
    stack = sorted(roots)
    visited: set[str] = set()
    while stack:
        nid = stack.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        ordered.append(nid)
        for child in sorted(children.get(nid, [])):
            if child not in visited:
                stack.append(child)

    for mid in msg_ids - visited:
        ordered.append(mid)

    messages = []
    for mid in ordered:
        m = nodes.get(mid, {})
        messages.append({
            "id": mid,
            "role": m.get("role", "?"),
            "text": m.get("text", ""),
        })

    return {
        "id": conversation_id,
        "message_count": len(messages),
        "messages": messages,
    }


@mcp.tool()
def get_message_context(
    message_id: str,
    depth: int = 2,
) -> dict[str, Any]:
    """
    Get a message with its surrounding reply chain context.

    Args:
        message_id: Message node ID.
        depth: How many ancestors and descendants to include (default 2).

    Returns:
        Dict with the target message, ancestors, and descendants.
    """
    g = _g()
    nodes = g["nodes"]
    edges_replies_to: set[tuple[str, str]] = g["edges_replies_to"]
    edges_contains: set[tuple[str, str]] = g["edges_contains"]

    if message_id not in nodes:
        return {"error": f"message not found: {message_id}"}

    parent_map: dict[str, str] = {child: parent for parent, child in edges_replies_to}
    children_map: dict[str, list[str]] = defaultdict(list)
    for parent, child in edges_replies_to:
        children_map[parent].append(child)

    msg_to_conv: dict[str, str] = {mid: cid for cid, mid in edges_contains}

    def _fmt(mid: str) -> dict[str, Any]:
        m = nodes.get(mid, {})
        return {"id": mid, "role": m.get("role", "?"), "text": m.get("text", "")}

    ancestors: list[dict[str, Any]] = []
    cur = message_id
    for _ in range(depth):
        p = parent_map.get(cur)
        if not p or nodes.get(p, {}).get("label") != "Message":
            break
        ancestors.insert(0, _fmt(p))
        cur = p

    descendants: list[dict[str, Any]] = []
    queue = [(message_id, 0)]
    while queue:
        nid, d = queue.pop(0)
        if d >= depth:
            continue
        for child in children_map.get(nid, []):
            descendants.append(_fmt(child))
            queue.append((child, d + 1))

    return {
        "conversation_id": msg_to_conv.get(message_id, ""),
        "ancestors": ancestors,
        "message": _fmt(message_id),
        "descendants": descendants,
    }


@mcp.tool()
def co_occurring_entities(
    entity_id: str,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Find entities that co-occur with a given entity across messages.

    Args:
        entity_id: Source entity node ID.
        limit: Maximum co-occurring entities to return (default 20).

    Returns:
        List of co-occurring entity dicts sorted by co-occurrence frequency.
    """
    g = _g()
    nodes = g["nodes"]
    edges_cooc: set[tuple[str, str]] = g["edges_cooc"]

    if entity_id not in nodes:
        return [{"error": f"entity not found: {entity_id}"}]

    peers = []
    for a, b in edges_cooc:
        other = None
        if a == entity_id:
            other = b
        elif b == entity_id:
            other = a
        if other:
            attrs = nodes.get(other, {})
            peers.append({
                "id": other,
                "name": attrs.get("name", ""),
                "entity_type": attrs.get("entity_type", ""),
            })

    return peers[:limit]


@mcp.tool()
def top_entities(
    entity_type: str = "",
    limit: int = 30,
) -> list[dict[str, Any]]:
    """
    Return the most-mentioned entities overall or by type.

    Args:
        entity_type: Optional spaCy NER label filter (PERSON, ORG, GPE, ...).
        limit: How many to return (default 30).

    Returns:
        Ranked list of entity dicts with name, entity_type, mention_count.
    """
    g = _g()
    nodes = g["nodes"]
    edges_mentions: set[tuple[str, str]] = g["edges_mentions"]

    mention_counts: Counter[str] = Counter(eid for _, eid in edges_mentions)

    results = []
    for node_id, attrs in nodes.items():
        if attrs.get("label") != "Entity":
            continue
        etype = attrs.get("entity_type", "")
        if entity_type and etype != entity_type.upper():
            continue
        results.append({
            "id": node_id,
            "name": attrs.get("name", ""),
            "entity_type": etype,
            "mention_count": mention_counts.get(node_id, 0),
        })

    results.sort(key=lambda x: -x["mention_count"])
    return results[:limit]


@mcp.tool()
def top_keywords(
    limit: int = 30,
) -> list[dict[str, Any]]:
    """
    Return the most frequent TF-IDF keywords across all messages.

    Args:
        limit: How many to return (default 30).

    Returns:
        Ranked list of keyword dicts with name, message_count, avg_tfidf.
    """
    g = _g()
    nodes = g["nodes"]
    edges_keywords: list[tuple[str, str, float]] = g["edges_keywords"]

    kw_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "total_score": 0.0}
    )
    for _msg_id, kw_id, score in edges_keywords:
        kw_stats[kw_id]["count"] += 1
        kw_stats[kw_id]["total_score"] += score

    results = []
    for node_id, attrs in nodes.items():
        if attrs.get("label") != "Keyword":
            continue
        stats = kw_stats.get(node_id, {"count": 0, "total_score": 0.0})
        count = stats["count"]
        avg = stats["total_score"] / count if count else 0.0
        results.append({
            "id": node_id,
            "name": attrs.get("name", ""),
            "message_count": count,
            "avg_tfidf": round(avg, 4),
        })

    results.sort(key=lambda x: (-x["message_count"], -x["avg_tfidf"]))
    return results[:limit]


@mcp.tool()
def search_message_text(
    query: str,
    role: str = "",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Full-text search over message text (substring, case-insensitive).

    Args:
        query: Text to search for within message bodies.
        role: Optional role filter: 'user' or 'assistant'.
        limit: Maximum results to return (default 10).

    Returns:
        List of matching message dicts with id, role, text_preview, conversation_id.
    """
    g = _g()
    nodes = g["nodes"]
    edges_contains: set[tuple[str, str]] = g["edges_contains"]

    msg_to_conv: dict[str, str] = {mid: cid for cid, mid in edges_contains}
    pattern = re.compile(re.escape(query), re.IGNORECASE)

    results: list[dict[str, Any]] = []
    for node_id, attrs in nodes.items():
        if attrs.get("label") != "Message":
            continue
        if role and attrs.get("role") != role:
            continue
        text = attrs.get("text", "")
        m = pattern.search(text)
        if not m:
            continue
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        snippet = ("..." if start else "") + text[start:end] + ("..." if end < len(text) else "")

        results.append({
            "id": node_id,
            "role": attrs.get("role", "?"),
            "snippet": snippet,
            "conversation_id": msg_to_conv.get(node_id, ""),
        })
        if len(results) >= limit:
            break

    return results


@mcp.tool()
def entity_timeline(
    entity_id: str,
    other_entity_id: str = "",
) -> dict[str, Any]:
    """
    Show which conversations mention an entity (or a pair of entities together).

    Useful for understanding when/where a topic appeared across your history.

    Args:
        entity_id: Primary entity node ID.
        other_entity_id: Optional second entity ID for intersection (co-mention at
                         conversation level).

    Returns:
        Dict with entity info and list of conversation IDs that mention it.
    """
    g = _g()
    nodes = g["nodes"]
    edges_mentions: set[tuple[str, str]] = g["edges_mentions"]
    edges_contains: set[tuple[str, str]] = g["edges_contains"]

    if entity_id not in nodes:
        return {"error": f"entity not found: {entity_id}"}

    msg_to_conv: dict[str, str] = {mid: cid for cid, mid in edges_contains}

    convs_a = {
        msg_to_conv[msg_id]
        for msg_id, eid in edges_mentions
        if eid == entity_id and msg_id in msg_to_conv
    }

    result: dict[str, Any] = {
        "entity": {
            "id": entity_id,
            "name": nodes[entity_id].get("name", ""),
            "entity_type": nodes[entity_id].get("entity_type", ""),
        },
        "conversation_count": len(convs_a),
        "conversation_ids": sorted(convs_a),
    }

    if other_entity_id:
        if other_entity_id not in nodes:
            result["error_b"] = f"second entity not found: {other_entity_id}"
            return result
        convs_b = {
            msg_to_conv[msg_id]
            for msg_id, eid in edges_mentions
            if eid == other_entity_id and msg_id in msg_to_conv
        }
        intersection = convs_a & convs_b
        result["other_entity"] = {
            "id": other_entity_id,
            "name": nodes[other_entity_id].get("name", ""),
            "entity_type": nodes[other_entity_id].get("entity_type", ""),
        }
        result["co_mentioned_in"] = sorted(intersection)
        result["co_mention_count"] = len(intersection)

    return result


# ── Entity centrality (betweenness bridge entities) ──────────────────


@mcp.tool()
def entity_centrality(
    entity_type: str = "",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """
    Find bridge entities via betweenness centrality on the entity co-occurrence graph.

    High betweenness means an entity connects otherwise-disconnected topic clusters.
    Computation uses sampled Brandes (k=500) on large graphs for speed.

    Args:
        entity_type: Optional spaCy label filter (PERSON, ORG, GPE, ...).
        limit: How many entities to return (default 20).

    Returns:
        List of entities ranked by betweenness, with name, type, score, degree.
    """
    g = _g()
    nodes = g["nodes"]
    edges_cooc: set[tuple[str, str]] = g["edges_cooc"]

    entity_ids = {eid for eid, attrs in nodes.items() if attrs.get("label") == "Entity"}

    cg = nx.Graph()
    cg.add_nodes_from(entity_ids)
    for a, b in edges_cooc:
        if a in entity_ids and b in entity_ids:
            cg.add_edge(a, b)

    n = cg.number_of_nodes()
    if n < 2:
        return []

    k = min(500, n)
    exact = n <= 500
    centrality = nx.betweenness_centrality(cg, k=None if exact else k, normalized=True, seed=42, endpoints=False)

    results = []
    for eid, score in centrality.items():
        n = nodes[eid]
        etype = n.get("entity_type", "")
        if entity_type and etype != entity_type.upper():
            continue
        results.append({
            "id": eid,
            "name": n.get("name", ""),
            "entity_type": etype,
            "betweenness": round(score, 6),
            "degree": cg.degree(eid),
        })

    results.sort(key=lambda x: -x["betweenness"])
    return results[:limit]


# ── Similar conversations (Jaccard similarity) ───────────────────────


def _conv_title(conv_id: str) -> str:
    msgs = _m()
    for m in msgs:
        if m.get("conversation_id") == conv_id:
            t = m.get("text", "")
            if isinstance(t, str) and t.strip():
                return t.strip()[:80]
    return conv_id[:16]


@mcp.tool()
def similar_conversations(
    conversation_id: str,
    threshold: float = 0.3,
    top: int = 10,
    include_keywords: bool = False,
) -> list[dict[str, Any]]:
    """
    Find conversations similar to a given conversation by Jaccard overlap.

    Compares entity sets (and optionally keyword sets) between the query
    conversation and all others. Requires messages.pkl for title previews.

    Args:
        conversation_id: Source conversation node ID.
        threshold: Minimum Jaccard score (default 0.3).
        top: Maximum results to return (default 10).
        include_keywords: Include TF-IDF keywords in similarity (default False).

    Returns:
        List of similar conversations with score, title, message_count.
    """
    g = _g()
    edges_contains: set[tuple[str, str]] = g["edges_contains"]
    edges_mentions: set[tuple[str, str]] = g["edges_mentions"]
    edges_keywords: list[tuple[str, str, float]] = g["edges_keywords"]

    msg_entities: dict[str, set[str]] = defaultdict(set)
    for msg_id, eid in edges_mentions:
        msg_entities[msg_id].add(eid)

    msg_keywords: dict[str, set[str]] = defaultdict(set)
    for msg_id, kid, _w in edges_keywords:
        msg_keywords[msg_id].add(kid)

    conv_msgs: dict[str, list[str]] = defaultdict(list)
    for cid, mid in edges_contains:
        conv_msgs[cid].append(mid)

    conv_entity_sets: dict[str, set[str]] = {}
    conv_keyword_sets: dict[str, set[str]] = {}
    for cid, mids in conv_msgs.items():
        ents: set[str] = set()
        kws: set[str] = set()
        for mid in mids:
            ents |= msg_entities.get(mid, set())
            kws |= msg_keywords.get(mid, set())
        conv_entity_sets[cid] = ents
        conv_keyword_sets[cid] = kws

    if conversation_id not in conv_entity_sets:
        return [{"error": f"conversation not found: {conversation_id}"}]

    query_ents = conv_entity_sets[conversation_id]
    query_kws = conv_keyword_sets.get(conversation_id, set())
    if not query_ents:
        return [{"error": "query conversation has no entities"}]

    scored: list[tuple[float, str]] = []
    for cid, ents in conv_entity_sets.items():
        if cid == conversation_id:
            continue
        if not ents:
            continue
        ent_j = len(query_ents & ents) / len(query_ents | ents)

        if include_keywords:
            kws = conv_keyword_sets.get(cid, set())
            kw_j = len(query_kws & kws) / len(query_kws | kws) if query_kws and kws else 0.0
            score = 0.7 * ent_j + 0.3 * kw_j
        else:
            score = ent_j

        if score >= threshold:
            scored.append((score, cid))

    scored.sort(key=lambda x: -x[0])

    results = []
    for score, cid in scored[:top]:
        results.append({
            "conversation_id": cid,
            "jaccard": round(score, 4),
            "title": _conv_title(cid),
            "message_count": len(conv_msgs.get(cid, [])),
        })

    return results


# ── Topic clusters (Louvain community detection) ─────────────────────


def _entity_name(nodes: dict[str, Any], eid: str) -> str:
    n = nodes.get(eid, {})
    if isinstance(n, dict):
        name = n.get("name")
        return str(name) if name else eid.split("::", 2)[-1]
    return eid.split("::", 2)[-1]


def _entity_type(nodes: dict[str, Any], eid: str) -> str:
    n = nodes.get(eid, {})
    if isinstance(n, dict):
        return str(n.get("entity_type", ""))
    return ""


@mcp.tool()
def topic_clusters(
    min_size: int = 3,
    top_entities: int = 10,
) -> list[dict[str, Any]]:
    """
    Discover topic clusters via Louvain community detection on the entity
    co-occurrence graph.

    Runs separately on each connected component. Each cluster includes its
    top entities by degree, entity type distribution, and characteristic
    TF-IDF keywords. Skips components smaller than min_size.

    Args:
        min_size: Minimum entities per cluster (default 3).
        top_entities: How many top entities to show per cluster (default 10).

    Returns:
        List of clusters, each with id, size, entity_type_distribution,
        top_entities list, and top_keywords.
    """
    g = _g()
    nodes = g["nodes"]
    edges_cooc: set[tuple[str, str]] = g["edges_cooc"]
    edges_mentions: set[tuple[str, str]] = g["edges_mentions"]
    edges_keywords: list[tuple[str, str, float]] = g["edges_keywords"]

    entity_ids = {nid for nid, attrs in nodes.items() if attrs.get("label") == "Entity"}

    cg = nx.Graph()
    cg.add_nodes_from(entity_ids)
    for a, b in edges_cooc:
        if a in entity_ids and b in entity_ids:
            cg.add_edge(a, b)

    if cg.number_of_nodes() < 2:
        return []

    components = sorted(nx.connected_components(cg), key=len, reverse=True)
    large = [c for c in components if len(c) >= 3]

    msg_keywords: dict[str, set[str]] = defaultdict(set)
    for msg_id, kid, _w in edges_keywords:
        msg_keywords[msg_id].add(kid)

    entity_msgs: dict[str, set[str]] = defaultdict(set)
    for msg_id, eid in edges_mentions:
        entity_msgs[eid].add(msg_id)

    result_clusters: list[dict[str, Any]] = []
    for comp in large:
        if len(comp) < min_size:
            continue
        sg = cg.subgraph(comp)
        try:
            comms = nx.community.louvain_communities(sg, seed=42)
        except Exception:
            comms = [frozenset(comp)]

        for comm in comms:
            if len(comm) < min_size:
                continue
            comm_g = cg.subgraph(comm)
            deg = dict(comm_g.degree())
            top = sorted(deg.items(), key=lambda x: -x[1])[:top_entities]

            type_counts: Counter[str] = Counter()
            for eid in comm:
                type_counts[_entity_type(nodes, eid)] += 1

            cluster_msgs: set[str] = set()
            for eid in comm:
                cluster_msgs |= entity_msgs.get(eid, set())

            kw_counter: Counter[str] = Counter()
            for mid in cluster_msgs:
                for kid in msg_keywords.get(mid, set()):
                    name = _entity_name(nodes, kid)
                    if name:
                        kw_counter[name] += 1

            result_clusters.append({
                "cluster_id": len(result_clusters) + 1,
                "entity_count": len(comm),
                "internal_edges": comm_g.number_of_edges(),
                "type_distribution": dict(type_counts.most_common(10)),
                "top_entities": [
                    {"name": _entity_name(nodes, eid), "degree": d, "entity_type": _entity_type(nodes, eid)}
                    for eid, d in top
                ],
                "top_keywords": [{"keyword": kw, "count": c} for kw, c in kw_counter.most_common(5)],
            })

    return result_clusters


# ── Reply-chain depth/branching per conversation ─────────────────────


@mcp.tool()
def reply_chain_stats(
    conversation_id: str,
) -> dict[str, Any]:
    """
    Analyze reply-chain depth and branching for a specific conversation.

    Requires REPLY_TO edges in the knowledge graph. Computes topological
    depth, branching factor, and depth distribution via DAG traversal.

    Args:
        conversation_id: Conversation node ID.

    Returns:
        Dict with max_depth, mean_depth, branching_factor, depth_distribution,
        and message_count_in_chain, or an error if the conversation has no
        reply chains.
    """
    g = _g()
    nodes = g["nodes"]
    edges_replies_to: set[tuple[str, str]] = g["edges_replies_to"]
    edges_contains: set[tuple[str, str]] = g["edges_contains"]

    if conversation_id not in nodes:
        return {"error": f"conversation not found: {conversation_id}"}

    all_msg_ids = {nid for nid, attrs in nodes.items() if attrs.get("label") == "Message"}
    conv_msg_ids = {mid for cid, mid in edges_contains if cid == conversation_id and mid in all_msg_ids}

    if not conv_msg_ids:
        return {"error": "conversation has no message nodes in graph"}

    reply_present = [m for m in conv_msg_ids if m in {c for _, c in edges_replies_to}]
    if not reply_present:
        return {"conversation_id": conversation_id, "max_depth": 0, "mean_depth": 0.0, "branching_factor": 0.0, "message_count_in_chain": 0, "depth_distribution": []}

    rg = nx.DiGraph()
    rg.add_edges_from((p, c) for p, c in edges_replies_to if p in conv_msg_ids and c in conv_msg_ids)

    try:
        topo = list(nx.topological_sort(rg))
    except nx.NetworkXUnfeasible:
        dag = nx.DiGraph(nx.algorithms.dag.transitive_reduction(nx.DiGraph(rg)))
        topo = list(nx.topological_sort(dag))
        rg = dag

    depth_map: dict[str, int] = {}
    for nid in topo:
        preds = list(rg.predecessors(nid))
        depth_map[nid] = 1 if not preds else max(depth_map[p] for p in preds) + 1

    depths = list(depth_map.values())
    if not depths:
        return {"conversation_id": conversation_id, "max_depth": 0, "mean_depth": 0.0, "branching_factor": 0.0, "message_count_in_chain": 0, "depth_distribution": []}

    children: defaultdict[str, int] = defaultdict(int)
    for p, _c in edges_replies_to:
        if p in conv_msg_ids:
            children[p] += 1
    non_leaf = [c for c in children.values() if c > 0]
    branch_factor = sum(non_leaf) / len(non_leaf) if non_leaf else 0.0

    depth_dist = sorted(Counter(depths).items())
    max_c = max(c for _, c in depth_dist)

    return {
        "conversation_id": conversation_id,
        "max_depth": max(depths),
        "mean_depth": round(sum(depths) / len(depths), 2),
        "branching_factor": round(branch_factor, 3),
        "message_count_in_chain": len(depths),
        "depth_distribution": [{"depth": d, "count": c, "bar": "#" * int(20 * c / max_c)} for d, c in depth_dist],
    }


# ── Entity temporal metrics (lifespan, bursts, activity) ──────────────


def _time_bucket_temporal(ts: float | None, window_days: int) -> str:
    if ts is None:
        return "unknown"
    dt = datetime.fromtimestamp(ts, tz=UTC)
    if window_days >= 365:
        return dt.strftime("%Y")
    if window_days >= 28:
        return dt.strftime("%Y-%m")
    if window_days >= 7:
        return dt.strftime("%Y-W%V")
    if window_days >= 1:
        ordinal = dt.toordinal() // window_days
        return f"W{ordinal}"
    return dt.strftime("%Y-%m-%d")


@mcp.tool()
def entity_temporal_metrics(
    entity_id: str,
    window_days: int = 30,
) -> dict[str, Any]:
    """
    Compute temporal activity metrics for a specific entity.

    Shows first/last seen, total mentions, active buckets, burstiness
    (coefficient of variation), and per-bucket mention counts.

    Args:
        entity_id: Entity node ID (from search_entities).
        window_days: Size of each time bucket in days (default 30).

    Returns:
        Dict with entity info, lifespan, burst metrics, and per-bucket
        mention time series, or an error if entity/messages not found.
    """
    g = _g()
    nodes = g["nodes"]
    edges_mentions: set[tuple[str, str]] = g["edges_mentions"]

    if entity_id not in nodes:
        return {"error": f"entity not found: {entity_id}"}

    msgs = _m()
    msg_timestamps: dict[str, float | None] = {}
    for m in msgs:
        msg_timestamps[m["id"]] = m.get("create_time")

    bucket_counts: Counter[str] = Counter()
    for msg_id, eid in edges_mentions:
        if eid != entity_id:
            continue
        ts = msg_timestamps.get(msg_id)
        bucket = _time_bucket_temporal(ts, window_days)
        if bucket != "unknown":
            bucket_counts[bucket] += 1

    if not bucket_counts:
        return {"error": "entity not found in any timestamped bucket (may need messages.pkl)"}

    sorted_buckets = sorted(bucket_counts)
    total = sum(bucket_counts.values())
    active = len(bucket_counts)
    first = sorted_buckets[0]
    last = sorted_buckets[-1]

    all_time_buckets: set[str] = set()
    if msgs:
        for m in msgs:
            ts = m.get("create_time")
            b = _time_bucket_temporal(ts, window_days)
            if b != "unknown":
                all_time_buckets.add(b)

    if all_time_buckets:
        total_window = len(all_time_buckets)
    else:
        total_window = len(sorted_buckets)

    mean_ = total / max(total_window, 1)
    if total_window > 1 and mean_ > 0:
        counts_list = [bucket_counts.get(b, 0) for b in sorted(all_time_buckets)] if all_time_buckets else list(bucket_counts.values())
        variance = sum((c - mean_) ** 2 for c in counts_list) / total_window
        cv = math.sqrt(variance) / mean_ if mean_ > 0 else 0.0
        bursts = sum(1 for c in counts_list if c > mean_ + 2.0 * math.sqrt(variance))
    else:
        cv = 0.0
        bursts = 0

    return {
        "entity": {"id": entity_id, "name": str(nodes[entity_id].get("name", "")), "entity_type": str(nodes[entity_id].get("entity_type", ""))},
        "first_bucket": first,
        "last_bucket": last,
        "total_mentions": total,
        "active_buckets": active,
        "total_time_buckets": total_window,
        "burstiness_cv": round(cv, 4),
        "burst_events": bursts,
        "bucket_timeline": [{"bucket": b, "mentions": bucket_counts[b]} for b in sorted_buckets],
    }


# ── Entity timeline buckets (top entities per time bucket) ────────────


def _time_bucket_timeline(ts: float | None, freq: str) -> str:
    if ts is None:
        return "unknown"
    dt = datetime.fromtimestamp(ts, tz=UTC)
    if freq == "day":
        return dt.strftime("%Y-%m-%d")
    if freq == "week":
        return dt.strftime("%Y-W%V")
    if freq == "year":
        return dt.strftime("%Y")
    return dt.strftime("%Y-%m")


@mcp.tool()
def entity_timeline_bucket(
    bucket: str = "",
    freq: str = "month",
    top: int = 10,
) -> list[dict[str, Any]]:
    """
    Show the most-mentioned entities in a specific time bucket.

    Useful for understanding what topics were active in a given month/week/day.
    If bucket is empty, returns the top entities across all time (overall
    entity ranking filtered by time coverage).

    Args:
        bucket: Time bucket string like "2025-01" (for freq=month),
                "2025-W03" (for freq=week), "2025-01-15" (for freq=day),
                or "2025" (for freq=year). Leave empty for all-time ranking.
        freq: Bucket frequency (year/month/week/day) — must match the
              bucket format you provide (default 'month').
        top: How many entities to return (default 10).

    Returns:
        List of entity dicts with name, type, mention_count in that bucket.
    """
    g = _g()
    nodes = g["nodes"]
    edges_mentions: set[tuple[str, str]] = g["edges_mentions"]

    msgs = _m()
    msg_timestamps: dict[str, float | None] = {}
    for m in msgs:
        msg_timestamps[m["id"]] = m.get("create_time")

    bucket_counts: dict[str, Counter[str]] = {}
    for msg_id, eid in edges_mentions:
        ts = msg_timestamps.get(msg_id)
        b = _time_bucket_timeline(ts, freq)
        if b not in bucket_counts:
            bucket_counts[b] = Counter[str]()
        bucket_counts[b][eid] += 1

    if bucket:
        counts = bucket_counts.get(bucket)
        if not counts:
            return [{"error": f"no data for bucket '{bucket}' (available: {', '.join(sorted(bucket_counts)[:10])})"}]
        results = []
        for eid, c in counts.most_common(top):
            n = nodes.get(eid, {})
            results.append({
                "name": n.get("name", ""),
                "entity_type": n.get("entity_type", ""),
                "mention_count": c,
            })
        return results

    overall: Counter[str] = Counter()
    for _bucket, bucket_counter in bucket_counts.items():
        overall.update(bucket_counter)
    overall_sorted = [eid for eid, _ in overall.most_common(top)]
    results = []
    for eid in overall_sorted:
        n = nodes.get(eid, {})
        results.append({
            "id": eid,
            "name": n.get("name", ""),
            "entity_type": n.get("entity_type", ""),
            "mention_count": overall[eid],
        })
    return results


# ── Entry point ──────────────────────────────────────────────────────


def run_serve(graph_path: str | None = None, messages_path: str | None = None) -> None:
    if FastMCP is None:
        print("Error: mcp package not installed. Run: pip install convo-tools[mcp]")
        return
    global _GRAPH_PATH, _MESSAGES_PATH
    if graph_path:
        _GRAPH_PATH = Path(graph_path)
    if messages_path:
        _MESSAGES_PATH = Path(messages_path)
    mcp.run()
