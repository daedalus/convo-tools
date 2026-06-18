from __future__ import annotations

import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from convo_tools._graph_db import GraphDB


def semantic_search(
    db: GraphDB,
    query: str,
    top: int = 10,
    use_embeddings: bool = True,
    use_derived: bool = True,
    use_keywords: bool = True,
    min_similarity: float = 0.0,
    fetch_text: bool = False,
) -> list[dict[str, Any]]:
    terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 1]
    text_limit = None if fetch_text else 500

    results: dict[str, dict[str, Any]] = {}

    msg_entities = db.get_message_entities()
    msg_keywords = db.get_message_keywords()
    entity_msgs = db.get_entity_messages()
    keyword_msgs: dict[str, set[str]] = defaultdict(set)
    for mid, kws in msg_keywords.items():
        for kid in kws:
            keyword_msgs[kid].add(mid)

    entity_scores: dict[str, float] = {}
    all_entity_ids = {eid for eids in msg_entities.values() for eid in eids}
    for eid in all_entity_ids:
        node = db.get_node(eid)
        if not node:
            continue
        name = node.get("name", "").lower()
        for term in terms:
            if term in name:
                entity_scores[eid] = entity_scores.get(eid, 0) + 1.0

    for eid, escore in entity_scores.items():
        for mid in entity_msgs.get(eid, set()):
            if mid not in results:
                node = db.get_node(mid)
                results[mid] = {
                    "id": mid,
                    "score": 0.0,
                    "role": node.get("role", "?") if node else "?",
                    "text": (node.get("text", "") if node else "")[:text_limit] if text_limit else (node.get("text", "") if node else ""),
                    "matched_entities": [],
                    "matched_keywords": [],
                    "source": "keyword",
                }
            results[mid]["score"] += escore
            results[mid]["matched_entities"].append(eid)

    if use_keywords:
        for kid in keyword_msgs:
            node = db.get_node(kid)
            if not node:
                continue
            name = node.get("name", "").lower()
            for term in terms:
                if term in name:
                    for mid in keyword_msgs[kid]:
                        if mid not in results:
                            mnode = db.get_node(mid)
                            results[mid] = {
                                "id": mid,
                                "score": 0.0,
                                "role": mnode.get("role", "?") if mnode else "?",
                                "text": (mnode.get("text", "") if mnode else "")[:text_limit] if text_limit else (mnode.get("text", "") if mnode else ""),
                                "matched_entities": [],
                                "matched_keywords": [],
                                "source": "keyword",
                            }
                        results[mid]["score"] += 1.0
                        results[mid]["matched_keywords"].append(kid)

    if use_derived:
        conv_ids = {r["id"] for r in db.get_all_nodes_by_label("Conversation")}
        conv_msgs = db.get_conv_msgs_map()
        entity_msgs_direct = db.get_entity_messages()

        for eid in entity_scores:
            topics = db.get_conversation_topics(eid)
            if topics:
                for conv_id, weight in topics:
                    for mid in conv_msgs.get(conv_id, []):
                        if mid not in results:
                            mnode = db.get_node(mid)
                            results[mid] = {
                                "id": mid,
                                "score": 0.0,
                                "role": mnode.get("role", "?") if mnode else "?",
                                "text": (mnode.get("text", "") if mnode else "")[:text_limit] if text_limit else (mnode.get("text", "") if mnode else ""),
                                "matched_entities": [],
                                "matched_keywords": [],
                                "source": "derived",
                            }
                        results[mid]["score"] += weight * 0.5

    if use_embeddings:
        try:
            from convo_tools._embeddings import load_embeddings, find_similar_nodes
            embeddings = load_embeddings(db)
            if embeddings:
                top_entity_ids = sorted(entity_scores.keys(), key=lambda e: -entity_scores[e])[:3]
                for eid in top_entity_ids:
                    if eid in embeddings:
                        similar = find_similar_nodes(db, eid, top=5, min_similarity=min_similarity)
                        for sim_id, sim_score in similar:
                            if sim_id.startswith("msg::"):
                                if sim_id not in results:
                                    mnode = db.get_node(sim_id)
                                    results[sim_id] = {
                                        "id": sim_id,
                                        "score": 0.0,
                                        "role": mnode.get("role", "?") if mnode else "?",
                                        "text": (mnode.get("text", "") if mnode else "")[:text_limit] if text_limit else (mnode.get("text", "") if mnode else ""),
                                        "matched_entities": [],
                                        "matched_keywords": [],
                                        "source": "embedding",
                                    }
                                results[sim_id]["score"] += sim_score * 0.3
        except Exception:
            pass

    sorted_results = sorted(results.values(), key=lambda x: -x["score"])
    return sorted_results[:top]


def get_entity_neighborhood(
    db: GraphDB,
    entity_id: str,
    depth: int = 1,
) -> dict[str, Any]:
    entity_msgs = db.get_entity_messages()
    msg_entities = db.get_message_entities()
    conv_msgs = db.get_conv_msgs_map()
    msg_to_conv = db.get_msg_to_conv_map()

    neighborhood: dict[str, Any] = {
        "entity": None,
        "direct_messages": [],
        "related_entities": {},
        "conversations": [],
    }

    node = db.get_node(entity_id)
    if not node:
        return neighborhood

    neighborhood["entity"] = {
        "id": entity_id,
        "name": node.get("name", ""),
        "entity_type": node.get("entity_type", ""),
        "mention_count": node.get("mention_count", 0),
        "domain": node.get("domain", ""),
    }

    msgs = entity_msgs.get(entity_id, set())
    for mid in msgs:
        mnode = db.get_node(mid)
        if mnode:
            neighborhood["direct_messages"].append({
                "id": mid,
                "role": mnode.get("role", "?"),
                "text": mnode.get("text", "")[:300],
                "conversation_id": msg_to_conv.get(mid, ""),
            })

    related: dict[str, int] = defaultdict(int)
    for mid in msgs:
        for other_eid in msg_entities.get(mid, set()):
            if other_eid != entity_id:
                related[other_eid] += 1

    top_related = sorted(related.items(), key=lambda x: -x[1])[:20]
    for reid, count in top_related:
        rnode = db.get_node(reid)
        if rnode:
            neighborhood["related_entities"][reid] = {
                "name": rnode.get("name", ""),
                "entity_type": rnode.get("entity_type", ""),
                "co_mentions": count,
            }

    conv_ids = set()
    for mid in msgs:
        cid = msg_to_conv.get(mid)
        if cid:
            conv_ids.add(cid)

    for cid in list(conv_ids)[:10]:
        cnode = db.get_node(cid)
        if cnode:
            meta = db.get_conv_meta(cid)
            neighborhood["conversations"].append({
                "id": cid,
                "message_count": len(conv_msgs.get(cid, [])),
                "dominant_entities": meta.get("dominant_entities", "") if meta else "",
            })

    return neighborhood


def traverse_from_entity(
    db: GraphDB,
    entity_id: str,
    max_hops: int = 2,
) -> dict[str, Any]:
    visited: set[str] = {entity_id}
    frontier = {entity_id}
    paths: list[list[str]] = []

    for hop in range(max_hops):
        next_frontier: set[str] = set()
        for eid in frontier:
            similar = db.find_similar_nodes(eid, top=5)
            for sim_id, sim_score in similar:
                if sim_id not in visited and sim_id.startswith("entity::"):
                    visited.add(sim_id)
                    next_frontier.add(sim_id)
                    paths.append([entity_id, sim_id])

            bridges = db.get_entity_bridges(eid, top=5)
            for bridge_id, path_len, betweenness in bridges:
                if bridge_id not in visited:
                    visited.add(bridge_id)
                    next_frontier.add(bridge_id)
                    paths.append([eid, bridge_id])

        frontier = next_frontier

    result_paths = []
    for path in paths:
        path_info = []
        for eid in path:
            node = db.get_node(eid)
            if node:
                path_info.append({
                    "id": eid,
                    "name": node.get("name", ""),
                    "entity_type": node.get("entity_type", ""),
                })
            else:
                path_info.append({"id": eid})
        result_paths.append(path_info)

    return {
        "start": entity_id,
        "reachable": len(visited) - 1,
        "paths": result_paths,
    }
