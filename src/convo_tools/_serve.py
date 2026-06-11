from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import networkx as nx

from convo_tools._graph_db import GraphDB

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    FastMCP: Any = None  # type: ignore[no-redef]


P = Path.home() / ".convo-tools"

_graph: GraphDB | None = None
_GRAPH_PATH: Path = P / "knowledge_graph.db"


def _load_graph() -> GraphDB:
    global _graph
    if _graph is None:
        if not _GRAPH_PATH.exists():
            raise FileNotFoundError(
                f"Graph DB not found: {_GRAPH_PATH}. "
                "Run: convo-tools -m graph messages.pkl"
            )
        _graph = GraphDB(_GRAPH_PATH)
    return _graph


def _g() -> GraphDB:
    return _load_graph()


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
    return _g().graph_stats()


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
    db = _g()
    mention_counts = db.get_entity_mention_counts()
    q = query.lower()
    results = []
    for node in db.get_all_nodes_by_label("Entity"):
        name = node.get("name", "")
        etype = node.get("entity_type", "")
        if q not in name:
            continue
        if entity_type and etype != entity_type.upper():
            continue
        results.append({
            "id": node["id"],
            "name": name,
            "entity_type": etype,
            "mention_count": mention_counts.get(node["id"], 0),
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
    db = _g()
    kw_stats = db.get_keyword_stats()
    q = query.lower()
    results = []
    for node in db.get_all_nodes_by_label("Keyword"):
        name = node.get("name", "")
        if q not in name:
            continue
        stats = kw_stats.get(node["id"], {"count": 0, "total_score": 0.0})
        count = stats["count"]
        avg = stats["total_score"] / count if count else 0.0
        results.append({
            "id": node["id"],
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
    db = _g()
    if not db.node_exists(entity_id):
        return [{"error": f"entity not found: {entity_id}"}]

    msg_to_conv = db.get_msg_to_conv_map()
    results = []
    for msg_id, eid in db.get_edges_mentions():
        if eid != entity_id:
            continue
        msg = db.get_node(msg_id) or {}
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
    db = _g()
    if not db.node_exists(keyword_id):
        return [{"error": f"keyword not found: {keyword_id}"}]

    msg_to_conv = db.get_msg_to_conv_map()
    hits = [
        (msg_id, score)
        for msg_id, kw_id, score in db.get_edges_keywords()
        if kw_id == keyword_id and score >= min_tfidf
    ]
    hits.sort(key=lambda x: -x[1])

    results = []
    for msg_id, score in hits[:limit]:
        msg = db.get_node(msg_id) or {}
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
    db = _g()
    conv = db.get_node(conversation_id)
    if conv is None:
        return {"error": f"conversation not found: {conversation_id}"}

    if not include_messages:
        return {"id": conversation_id, **conv}

    edges_contains = db.get_edges_contains()
    edges_replies_to = db.get_edges_replies_to()

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
        m = db.get_node(mid) or {}
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
    db = _g()
    if not db.node_exists(message_id):
        return {"error": f"message not found: {message_id}"}

    edges_replies_to = db.get_edges_replies_to()
    msg_to_conv = db.get_msg_to_conv_map()

    parent_map: dict[str, str] = {child: parent for parent, child in edges_replies_to}
    children_map: dict[str, list[str]] = defaultdict(list)
    for parent, child in edges_replies_to:
        children_map[parent].append(child)

    def _fmt(mid: str) -> dict[str, Any]:
        m = db.get_node(mid) or {}
        return {"id": mid, "role": m.get("role", "?"), "text": m.get("text", "")}

    ancestors: list[dict[str, Any]] = []
    cur = message_id
    for _ in range(depth):
        p = parent_map.get(cur)
        if not p:
            break
        p_node = db.get_node(p)
        if not p_node or p_node.get("label") != "Message":
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
    db = _g()
    if not db.node_exists(entity_id):
        return [{"error": f"entity not found: {entity_id}"}]

    peers = []
    for a, b, w in db.get_edges_cooc():
        other = None
        if a == entity_id:
            other = b
        elif b == entity_id:
            other = a
        if other:
            n = db.get_node(other) or {}
            peers.append({
                "id": other,
                "name": n.get("name", ""),
                "entity_type": n.get("entity_type", ""),
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
    db = _g()
    mention_counts = db.get_entity_mention_counts()

    results = []
    for node in db.get_all_nodes_by_label("Entity"):
        etype = node.get("entity_type", "")
        if entity_type and etype != entity_type.upper():
            continue
        results.append({
            "id": node["id"],
            "name": node.get("name", ""),
            "entity_type": etype,
            "mention_count": mention_counts.get(node["id"], 0),
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
    db = _g()
    kw_stats = db.get_keyword_stats()

    results = []
    for node in db.get_all_nodes_by_label("Keyword"):
        stats = kw_stats.get(node["id"], {"count": 0, "total_score": 0.0})
        count = stats["count"]
        avg = stats["total_score"] / count if count else 0.0
        results.append({
            "id": node["id"],
            "name": node.get("name", ""),
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
    db = _g()
    msg_to_conv = db.get_msg_to_conv_map()
    pattern = re.compile(re.escape(query), re.IGNORECASE)

    results: list[dict[str, Any]] = []
    for node in db.get_all_nodes_by_label("Message"):
        if role and node.get("role") != role:
            continue
        text = node.get("text", "")
        m = pattern.search(text)
        if not m:
            continue
        start = max(0, m.start() - 80)
        end = min(len(text), m.end() + 80)
        snippet = ("..." if start else "") + text[start:end] + ("..." if end < len(text) else "")

        results.append({
            "id": node["id"],
            "role": node.get("role", "?"),
            "snippet": snippet,
            "conversation_id": msg_to_conv.get(node["id"], ""),
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
    db = _g()
    entity = db.get_node(entity_id)
    if entity is None:
        return {"error": f"entity not found: {entity_id}"}

    edges_mentions = db.get_edges_mentions()
    msg_to_conv = db.get_msg_to_conv_map()

    convs_a = {
        msg_to_conv[msg_id]
        for msg_id, eid in edges_mentions
        if eid == entity_id and msg_id in msg_to_conv
    }

    result: dict[str, Any] = {
        "entity": {
            "id": entity_id,
            "name": entity.get("name", ""),
            "entity_type": entity.get("entity_type", ""),
        },
        "conversation_count": len(convs_a),
        "conversation_ids": sorted(convs_a),
    }

    if other_entity_id:
        other = db.get_node(other_entity_id)
        if other is None:
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
            "name": other.get("name", ""),
            "entity_type": other.get("entity_type", ""),
        }
        result["co_mentioned_in"] = sorted(intersection)
        result["co_mention_count"] = len(intersection)

    return result


@mcp.tool()
def entity_centrality(
    entity_type: str = "",
    limit: int = 20,
    min_weight: int = 2,
) -> list[dict[str, Any]]:
    """
    Find bridge entities via betweenness centrality on the entity co-occurrence graph.

    High betweenness means an entity connects otherwise-disconnected topic clusters.
    Computation uses sampled Brandes (k=500) on large graphs for speed.

    Args:
        entity_type: Optional spaCy label filter (PERSON, ORG, GPE, ...).
        limit: How many entities to return (default 20).
        min_weight: Minimum co-occurrence weight (default 2; higher = fewer edges, less noise).

    Returns:
        List of entities ranked by betweenness, with name, type, score, degree.
    """
    db = _g()
    cg = db.build_entity_cooc_graph(min_weight=min_weight)

    n = cg.number_of_nodes()
    if n < 2:
        return []

    k = min(500, n)
    exact = n <= 500
    centrality = nx.betweenness_centrality(cg, k=None if exact else k, normalized=True, seed=42, endpoints=False)

    results = []
    for eid, score in centrality.items():
        node = db.get_node(eid) or {}
        etype = node.get("entity_type", "")
        if entity_type and etype != entity_type.upper():
            continue
        results.append({
            "id": eid,
            "name": node.get("name", ""),
            "entity_type": etype,
            "betweenness": round(score, 6),
            "degree": cg.degree(eid),
        })

    results.sort(key=lambda x: -x["betweenness"])
    return results[:limit]


def _conv_title(conv_id: str) -> str:
    db = _g()
    edges_contains = db.get_edges_contains()
    mids = [mid for cid, mid in edges_contains if cid == conv_id]
    for mid in mids:
        msg = db.get_node(mid)
        if msg:
            t = msg.get("text", "")
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
    conversation and all others.

    Args:
        conversation_id: Source conversation node ID.
        threshold: Minimum Jaccard score (default 0.3).
        top: Maximum results to return (default 10).
        include_keywords: Include TF-IDF keywords in similarity (default False).

    Returns:
        List of similar conversations with score, title, message_count.
    """
    db = _g()
    edges_mentions = db.get_edges_mentions()
    edges_keywords = db.get_edges_keywords()
    edges_contains = db.get_edges_contains()

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


def _entity_name(db: GraphDB, eid: str) -> str:
    n = db.get_node(eid) or {}
    name = n.get("name")
    return str(name) if name else eid.split("::", 2)[-1]


def _entity_type(db: GraphDB, eid: str) -> str:
    n = db.get_node(eid) or {}
    return str(n.get("entity_type", ""))


@mcp.tool()
def topic_clusters(
    min_size: int = 3,
    top_entities: int = 10,
    min_weight: int = 2,
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
        min_weight: Minimum co-occurrence weight (default 2; higher = fewer edges, less noise).

    Returns:
        List of clusters, each with id, size, entity_type_distribution,
        top_entities list, and top_keywords.
    """
    db = _g()
    cg = db.build_entity_cooc_graph(min_weight=min_weight)

    if cg.number_of_nodes() < 2:
        return []

    entity_msgs = db.get_entity_messages()
    msg_keywords = db.get_message_keywords()

    components = sorted(nx.connected_components(cg), key=len, reverse=True)
    large = [c for c in components if len(c) >= 3]

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
                type_counts[_entity_type(db, eid)] += 1

            cluster_msgs: set[str] = set()
            for eid in comm:
                cluster_msgs |= entity_msgs.get(eid, set())

            kw_counter: Counter[str] = Counter()
            for mid in cluster_msgs:
                for kid in msg_keywords.get(mid, set()):
                    name = _entity_name(db, kid)
                    if name:
                        kw_counter[name] += 1

            result_clusters.append({
                "cluster_id": len(result_clusters) + 1,
                "entity_count": len(comm),
                "internal_edges": comm_g.number_of_edges(),
                "type_distribution": dict(type_counts.most_common(10)),
                "top_entities": [
                    {"name": _entity_name(db, eid), "degree": d, "entity_type": _entity_type(db, eid)}
                    for eid, d in top
                ],
                "top_keywords": [{"keyword": kw, "count": c} for kw, c in kw_counter.most_common(5)],
            })

    return result_clusters


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
    db = _g()
    conv = db.get_node(conversation_id)
    if conv is None:
        return {"error": f"conversation not found: {conversation_id}"}

    edges_replies_to = db.get_edges_replies_to()
    edges_contains = db.get_edges_contains()
    all_msg_ids = db.get_message_id_set()
    conv_msg_ids = {mid for cid, mid in edges_contains if cid == conversation_id and mid in all_msg_ids}

    if not conv_msg_ids:
        return {"error": "conversation has no message nodes in graph"}

    reply_present = [m for m in conv_msg_ids if m in {c for _, c in edges_replies_to}]
    if not reply_present:
        return {
            "conversation_id": conversation_id,
            "max_depth": 0,
            "mean_depth": 0.0,
            "branching_factor": 0.0,
            "message_count_in_chain": 0,
            "depth_distribution": [],
        }

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
        return {
            "conversation_id": conversation_id,
            "max_depth": 0,
            "mean_depth": 0.0,
            "branching_factor": 0.0,
            "message_count_in_chain": 0,
            "depth_distribution": [],
        }

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
        "depth_distribution": [
            {"depth": d, "count": c, "bar": "#" * int(20 * c / max_c)} for d, c in depth_dist
        ],
    }


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
    db = _g()
    entity = db.get_node(entity_id)
    if entity is None:
        return {"error": f"entity not found: {entity_id}"}

    timestamps = db.get_message_timestamps()
    if not timestamps:
        return {
            "entity": {
                "id": entity_id,
                "name": entity.get("name", ""),
                "entity_type": entity.get("entity_type", ""),
            },
            "error": "no timestamps in graph; rebuild with: convo-tools -m graph ...",
        }

    edges_mentions = db.get_edges_mentions()
    bucket_ents: defaultdict[str, Counter[str]] = defaultdict(Counter)

    for msg_id, ent_id in edges_mentions:
        ts = timestamps.get(msg_id)
        bucket = _time_bucket_temporal(ts, window_days)
        if bucket != "unknown":
            bucket_ents[bucket][ent_id] += 1

    sorted_buckets = sorted(b for b in bucket_ents)
    if not sorted_buckets:
        return {
            "entity": {
                "id": entity_id,
                "name": entity.get("name", ""),
                "entity_type": entity.get("entity_type", ""),
            },
            "error": "no timestamped mentions found for this entity",
        }

    bucket_cts: list[tuple[str, int]] = []
    for b in sorted_buckets:
        count = bucket_ents[b].get(entity_id, 0)
        if count:
            bucket_cts.append((b, count))

    if not bucket_cts:
        return {
            "entity": {
                "id": entity_id,
                "name": entity.get("name", ""),
                "entity_type": entity.get("entity_type", ""),
            },
            "total_mentions": 0,
            "active_buckets": 0,
            "message": "entity not found in any timestamped bucket",
        }

    first_bucket = bucket_cts[0][0]
    last_bucket = bucket_cts[-1][0]
    total = sum(c for _, c in bucket_cts)
    n_active = len(bucket_cts)
    mention_days = [c for _, c in bucket_cts]

    mean_ = total / max(len(sorted_buckets), 1)
    if len(sorted_buckets) > 1 and mean_ > 0:
        variance = sum((c - mean_) ** 2 for c in mention_days) / len(sorted_buckets)
        bursts = sum(1 for c in mention_days if mean_ > 0 and c > mean_ + 2.0 * math.sqrt(variance))
        cv = math.sqrt(variance) / mean_
    else:
        bursts = 0
        cv = 0.0

    return {
        "entity": {
            "id": entity_id,
            "name": entity.get("name", ""),
            "entity_type": entity.get("entity_type", ""),
        },
        "first_bucket": first_bucket,
        "last_bucket": last_bucket,
        "total_mentions": total,
        "active_buckets": n_active,
        "burst_count": bursts,
        "cv": round(cv, 4),
        "per_bucket": [{"bucket": b, "count": c} for b, c in bucket_cts],
    }


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
    db = _g()
    edges_mentions = db.get_edges_mentions()

    if not bucket:
        mention_counts: Counter[str] = Counter(eid for _, eid in edges_mentions)
        results = []
        for eid, c in mention_counts.most_common(top):
            n = db.get_node(eid) or {}
            results.append({
                "id": eid,
                "name": n.get("name", ""),
                "entity_type": n.get("entity_type", ""),
                "mention_count": c,
            })
        return results

    timestamps = db.get_message_timestamps()
    if not timestamps:
        return [{"error": f"no data for bucket '{bucket}' (no timestamps in graph; rebuild with: convo-tools -m graph ...)"}]

    bucket_counts: Counter[str] = Counter()
    for msg_id, ent_id in edges_mentions:
        ts = timestamps.get(msg_id)
        if ts is None:
            continue
        ts_bucket = _time_bucket_timeline(ts, freq)
        if ts_bucket == bucket:
            bucket_counts[ent_id] += 1

    if not bucket_counts:
        return [{"error": f"no data for bucket '{bucket}'"}]

    results = []
    for eid, c in bucket_counts.most_common(top):
        n = db.get_node(eid) or {}
        results.append({
            "id": eid,
            "name": n.get("name", ""),
            "entity_type": n.get("entity_type", ""),
            "mention_count": c,
        })
    return results


def run_serve(graph_path: str | None = None) -> None:
    if FastMCP is None:
        print("Error: mcp package not installed. Run: pip install convo-tools[mcp]")
        return
    global _GRAPH_PATH
    if graph_path:
        _GRAPH_PATH = Path(graph_path)
    mcp.run()
