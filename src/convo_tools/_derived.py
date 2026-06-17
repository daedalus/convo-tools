from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

from convo_tools._graph_db import GraphDB
from convo_tools._util import _progressbar

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
    import igraph as ig

    entity_ids = db.get_entity_id_set()
    if len(entity_ids) < 4:
        return 0

    id_to_idx = {eid: i for i, eid in enumerate(sorted(entity_ids))}
    idx_to_id = {i: eid for eid, i in id_to_idx.items()}

    print("  Loading co-occurrence edges...", end="", flush=True)
    edges: list[tuple[int, int]] = []
    weights: list[int] = []
    for r in db._conn().execute(
        "SELECT a.entity_id AS entity_a, b.entity_id AS entity_b, weight "
        "FROM edge_cooc "
        "JOIN entity_int a ON edge_cooc.entity_a_int = a.int_id "
        "JOIN entity_int b ON edge_cooc.entity_b_int = b.int_id "
        "WHERE weight >= 2"
    ):
        a_idx = id_to_idx.get(r["entity_a"])
        b_idx = id_to_idx.get(r["entity_b"])
        if a_idx is not None and b_idx is not None:
            edges.append((a_idx, b_idx))
            weights.append(r["weight"])
    print(f" {len(edges)} edges")

    if not edges:
        return 0

    g = ig.Graph(n=len(entity_ids), edges=edges, directed=False)
    g.es["weight"] = weights

    components = g.connected_components()
    large_components = sorted([c for c in components if len(c) >= 3], key=len, reverse=True)

    import time as _time
    betweenness: list[float] = [0.0] * g.vcount()
    total_nodes = sum(len(c) for c in large_components)
    processed = 0

    for comp_idx, comp in enumerate(large_components):
        n = len(comp)
        bar = int(40 * processed / total_nodes) if total_nodes else 0
        print(f"\r  betweenness [{'#' * bar}{'.' * (40 - bar)}] {processed}/{total_nodes} nodes  ", end="", flush=True)

        sg = g.subgraph(comp)
        _t0 = _time.monotonic()
        try:
            cutoff = 4
            sg_betweenness = sg.betweenness(cutoff=cutoff)
        except Exception:
            processed += n
            continue
        elapsed = _time.monotonic() - _t0
        rate = n / elapsed if elapsed > 0 else 0
        print(f"\r  betweenness [{'#' * (int(40 * (processed + n) / total_nodes) if total_nodes else 0)}{'.' * (40 - int(40 * (processed + n) / total_nodes) if total_nodes else 0)}] {processed + n}/{total_nodes} nodes  ({rate:.0f} n/s)  ", end="", flush=True)

        for i, v_idx in enumerate(comp):
            betweenness[v_idx] = sg_betweenness[i]
        processed += n

    print(f"\r  betweenness [{'#' * 40}] {processed}/{total_nodes} nodes                    ")

    bridge_indices = {
        i for i, score in enumerate(betweenness) if score >= min_betweenness
    }

    if not bridge_indices:
        return 0

    conn = db._conn()
    _init_derived_schema(conn)
    count = 0

    bridge_entities_in_large = [
        v for comp in large_components for v in comp if v in bridge_indices
    ]
    total = len(bridge_entities_in_large)
    processed = 0

    for component in large_components:
        sub = g.subgraph(component)
        for v_idx in component:
            if v_idx not in bridge_indices:
                continue
            processed += 1
            if total > 0:
                bar_filled = int(40 * processed / total)
                bar = "#" * bar_filled + "." * (40 - bar_filled)
                print(f"\r  bridges [{'#' * bar_filled}{'.' * (40 - bar_filled)}] {processed}/{total}", end="", flush=True)
                continue
            try:
                target_distances = sub.shortest_paths(source=v_idx, cutoff=max_path_length)[0]
                for t_local, length in enumerate(target_distances):
                    if length is None or length == 0 or length > max_path_length:
                        continue
                    target_global = component[t_local]
                    if target_global == v_idx:
                        continue
                    if length >= 2:
                        b_a = betweenness[v_idx]
                        b_b = betweenness[target_global]
                        bridge_score = (b_a + b_b) / 2.0
                        conn.execute(
                            "INSERT OR REPLACE INTO edge_entity_bridge "
                            "(entity_a, entity_b, path_length, betweenness) "
                            "VALUES (?, ?, ?, ?)",
                            (idx_to_id[v_idx], idx_to_id[target_global], int(length), bridge_score),
                        )
                        count += 1
            except Exception:
                continue

    if total > 0:
        print()

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
