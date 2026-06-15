from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from pathlib import Path

from convo_tools._graph_db import GraphDB

DERIVED_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS edge_conversation_topic (
    conv_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (conv_id, entity_id)
);

CREATE TABLE IF NOT EXISTS edge_cross_message (
    msg_a TEXT NOT NULL,
    msg_b TEXT NOT NULL,
    similarity REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (msg_a, msg_b)
);

CREATE TABLE IF NOT EXISTS edge_entity_bridge (
    entity_a TEXT NOT NULL,
    entity_b TEXT NOT NULL,
    path_length INTEGER NOT NULL,
    betweenness REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (entity_a, entity_b)
);

CREATE INDEX IF NOT EXISTS idx_conv_topic_conv ON edge_conversation_topic(conv_id);
CREATE INDEX IF NOT EXISTS idx_conv_topic_entity ON edge_conversation_topic(entity_id);
CREATE INDEX IF NOT EXISTS idx_cross_msg_a ON edge_cross_message(msg_a);
CREATE INDEX IF NOT EXISTS idx_cross_msg_b ON edge_cross_message(msg_b);
CREATE INDEX IF NOT EXISTS idx_entity_bridge_a ON edge_entity_bridge(entity_a);
CREATE INDEX IF NOT EXISTS idx_entity_bridge_b ON edge_entity_bridge(entity_b);
"""


def _init_derived_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(DERIVED_SCHEMA_SQL)
    conn.commit()


def derive_conversation_topics(
    db: GraphDB,
    min_message_fraction: float = 0.3,
    min_mentions: int = 2,
) -> int:
    conv_msgs = db.get_conv_msgs_map()
    entity_msgs = db.get_entity_messages()

    entity_to_conv_mentions: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for eid, mids in entity_msgs.items():
        for mid in mids:
            conv_id = conv_msgs.get(mid)
            if conv_id:
                entity_to_conv_mentions[eid][conv_id] += 1

    conn = db._conn()
    _init_derived_schema(conn)
    count = 0

    for eid, conv_counts in entity_to_conv_mentions.items():
        for conv_id, mention_count in conv_counts.items():
            total_msgs = len(conv_msgs.get(conv_id, []))
            if total_msgs == 0:
                continue
            fraction = mention_count / total_msgs
            if fraction >= min_message_fraction and mention_count >= min_mentions:
                conn.execute(
                    "INSERT OR REPLACE INTO edge_conversation_topic (conv_id, entity_id, weight) "
                    "VALUES (?, ?, ?)",
                    (conv_id, eid, fraction),
                )
                count += 1

    conn.commit()
    return count


def derive_cross_message_links(
    db: GraphDB,
    jaccard_threshold: float = 0.3,
    max_comparisons: int = 100_000,
) -> int:
    msg_entities = db.get_message_entities()
    msg_keywords = db.get_message_keywords()

    msg_fingerprints: dict[str, set[str]] = {}
    for mid in set(msg_entities.keys()) | set(msg_keywords.keys()):
        fps = set()
        fps.update(msg_entities.get(mid, set()))
        fps.update(msg_keywords.get(mid, set()))
        if fps:
            msg_fingerprints[mid] = fps

    msg_ids = sorted(msg_fingerprints.keys())
    if not msg_ids:
        return 0

    inverted_index: dict[str, set[str]] = defaultdict(set)
    for mid, fps in msg_fingerprints.items():
        for fp in fps:
            inverted_index[fp].add(mid)

    conn = db._conn()
    _init_derived_schema(conn)

    candidate_pairs: set[tuple[str, str]] = set()
    for fp, mids in inverted_index.items():
        mids_list = sorted(mids)
        for i in range(len(mids_list)):
            for j in range(i + 1, len(mids_list)):
                pair = (mids_list[i], mids_list[j])
                candidate_pairs.add(pair)
                if len(candidate_pairs) >= max_comparisons:
                    break
            if len(candidate_pairs) >= max_comparisons:
                break
        if len(candidate_pairs) >= max_comparisons:
            break

    count = 0
    for a, b in candidate_pairs:
        fps_a = msg_fingerprints[a]
        fps_b = msg_fingerprints[b]
        intersection = len(fps_a & fps_b)
        union = len(fps_a | fps_b)
        if union == 0:
            continue
        similarity = intersection / union
        if similarity >= jaccard_threshold:
            conn.execute(
                "INSERT OR REPLACE INTO edge_cross_message (msg_a, msg_b, similarity) "
                "VALUES (?, ?, ?)",
                (a, b, similarity),
            )
            count += 1

    conn.commit()
    return count


def derive_entity_bridges(
    db: GraphDB,
    max_path_length: int = 4,
    min_betweenness: float = 0.01,
) -> int:
    g = db.build_entity_cooc_graph(min_weight=2)
    if g.number_of_nodes() < 4:
        return 0

    try:
        centrality = nx.betweenness_centrality(g, normalized=True, seed=42)
    except Exception:
        return 0

    bridge_entities = {
        eid for eid, score in centrality.items() if score >= min_betweenness
    }

    if not bridge_entities:
        return 0

    conn = db._conn()
    _init_derived_schema(conn)
    count = 0

    components = list(nx.connected_components(g))
    for component in components:
        if len(component) < 3:
            continue
        sub = g.subgraph(component)
        for eid in component:
            if eid not in bridge_entities:
                continue
            try:
                lengths = nx.single_source_shortest_path_length(sub, eid, cutoff=max_path_length)
                for target, length in lengths.items():
                    if target != eid and length >= 2:
                        b_a = centrality.get(eid, 0.0)
                        b_b = centrality.get(target, 0.0)
                        bridge_score = (b_a + b_b) / 2.0
                        conn.execute(
                            "INSERT OR REPLACE INTO edge_entity_bridge "
                            "(entity_a, entity_b, path_length, betweenness) "
                            "VALUES (?, ?, ?, ?)",
                            (eid, target, length, bridge_score),
                        )
                        count += 1
            except nx.NetworkXError:
                continue

    conn.commit()
    return count


def get_conversation_topics(db: GraphDB, conv_id: str) -> list[tuple[str, float]]:
    conn = db._conn()
    _init_derived_schema(conn)
    rows = conn.execute(
        "SELECT entity_id, weight FROM edge_conversation_topic "
        "WHERE conv_id = ? ORDER BY weight DESC",
        (conv_id,),
    ).fetchall()
    return [(r["entity_id"], r["weight"]) for r in rows]


def get_similar_messages(
    db: GraphDB, msg_id: str, top: int = 10
) -> list[tuple[str, float]]:
    conn = db._conn()
    _init_derived_schema(conn)
    rows = conn.execute(
        "SELECT msg_b as other, similarity FROM edge_cross_message WHERE msg_a = ? "
        "UNION "
        "SELECT msg_a as other, similarity FROM edge_cross_message WHERE msg_b = ? "
        "ORDER BY similarity DESC LIMIT ?",
        (msg_id, msg_id, top),
    ).fetchall()
    return [(r["other"], r["similarity"]) for r in rows]


def get_entity_bridges(
    db: GraphDB, entity_id: str, top: int = 10
) -> list[tuple[str, int, float]]:
    conn = db._conn()
    _init_derived_schema(conn)
    rows = conn.execute(
        "SELECT entity_b, path_length, betweenness FROM edge_entity_bridge "
        "WHERE entity_a = ? ORDER BY betweenness DESC LIMIT ?",
        (entity_id, top),
    ).fetchall()
    return [(r["entity_b"], r["path_length"], r["betweenness"]) for r in rows]


def derive_all(db: GraphDB) -> dict[str, int]:
    print("Deriving conversation topics...")
    n = derive_conversation_topics(db)
    print(f"  {n} conversation-topic edges")

    print("Deriving cross-message links...")
    n = derive_cross_message_links(db)
    print(f"  {n} cross-message links")

    print("Deriving entity bridges...")
    n = derive_entity_bridges(db)
    print(f"  {n} entity bridge edges")

    return {
        "conversation_topics": n,
        "cross_message_links": n,
        "entity_bridges": n,
    }
