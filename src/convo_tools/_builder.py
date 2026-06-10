from __future__ import annotations

import gc
import itertools
import pickle
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any

from convo_tools._graph_db import GraphDB
from convo_tools._util import _progressbar, _rss_mb

TOP_KEYWORDS_PER_MESSAGE = 5


def build_graph_to_db(
    all_messages: list[dict[str, Any]],
    db: GraphDB,
    debug: bool = False,
    known_message_ids: set[str] | None = None,
) -> None:
    import spacy  # noqa: PLC0415

    nlp = spacy.load(
        "en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"]
    )
    nlp.max_length = 100_000
    print(f"  RSS after spaCy load: {_rss_mb():.0f} MB")

    existing_known_ids: set[str] = known_message_ids or set()
    new_msg_ids: set[str] = {m["id"] for m in all_messages}
    all_known_ids: set[str] = existing_known_ids | new_msg_ids
    gc.collect()

    # ── Add conversation + message nodes/edges ──
    conv_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for msg in all_messages:
        conv_groups[msg["conversation_id"]].append(msg)

    conn = db._conn()
    node_count = 0
    edge_count = 0

    conn.execute("BEGIN TRANSACTION")
    try:
        for conv_id, msgs in conv_groups.items():
            db.upsert_node(conv_id, label="Conversation")
            node_count += 1
            for msg in msgs:
                db.upsert_node(
                    msg["id"],
                    label="Message",
                    role=msg["role"],
                    text=msg["text"][:1000],
                )
                node_count += 1
                db.add_edge_contains(conv_id, msg["id"])
                edge_count += 1
                if msg["parent"] and msg["parent"] in all_known_ids:
                    db.add_edge_replies_to(msg["parent"], msg["id"])
                    edge_count += 1
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    print(f"Graph: {node_count} nodes, {edge_count} edges")
    print(f"  RSS after message nodes: {_rss_mb():.0f} MB")
    gc.collect()

    # ── ENTITY EXTRACTION ──
    entity_count = 0
    print(f"\nExtracting entities from {len(all_messages)} messages...")

    max_len = nlp.max_length
    cooc_edges = 0

    n_batches = (len(all_messages) + 15) // 16
    batch_starts = list(range(0, len(all_messages), 16))
    for batch_start in _progressbar(
        batch_starts, n_batches, prefix="  entities ", width=40
    ):
        batch = all_messages[batch_start : batch_start + 16]

        seen_per_msg: dict[int, set[str]] = defaultdict(set)
        batch_mentions: set[tuple[str, str]] = set()
        batch_cooc: set[tuple[str, str]] = set()
        batch_nodes: dict[str, dict[str, Any]] = {}

        for offset, msg in enumerate(batch):
            text = msg["text"]
            if len(text) > max_len:
                chunks = [text[s : s + max_len] for s in range(0, len(text), max_len)]
            else:
                chunks = [text]

            msg_entities_list: list[str] = []

            for chunk in chunks:
                doc = nlp(chunk)
                for ent in doc.ents:
                    entity_id = f"entity::{ent.label_}::{ent.text.lower()}"
                    if entity_id not in batch_nodes:
                        batch_nodes[entity_id] = {
                            "label": "Entity",
                            "name": ent.text.lower(),
                            "entity_type": ent.label_,
                        }
                    if entity_id not in seen_per_msg[offset]:
                        batch_mentions.add((msg["id"], entity_id))
                        seen_per_msg[offset].add(entity_id)
                        entity_count += 1
                        msg_entities_list.append(f"{ent.label_}:{ent.text}")

            if debug and msg_entities_list:
                print(
                    f"  [{msg['role']}] {msg['id'][:8]}: {', '.join(msg_entities_list)}"
                )

        gc.collect()

        for entities in seen_per_msg.values():
            entity_list = list(entities)
            for a, b in itertools.combinations(entity_list, 2):
                if a > b:
                    a, b = b, a
                if (a, b) not in batch_cooc:
                    batch_cooc.add((a, b))
                    cooc_edges += 1

        # Write this batch to SQLite (batch methods manage their own transactions)
        if batch_nodes:
            db.add_nodes_batch(batch_nodes)
        if batch_mentions:
            db.add_mentions_batch(batch_mentions)
        if batch_cooc:
            db.add_cooc_batch(batch_cooc)

    gc.collect()
    print(f"  entities: done ({entity_count} total)")
    print(f"  Added {cooc_edges} co-occurrence edges")
    print(f"  RSS after entity extraction: {_rss_mb():.0f} MB")

    # ── KEYWORD EXTRACTION ──
    if all_messages:
        print("\nExtracting TF-IDF keywords...")
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )
        x_mat = vectorizer.fit_transform(m["text"] for m in all_messages)
        feature_names = vectorizer.get_feature_names_out()

        conn.execute("BEGIN TRANSACTION")
        try:
            for row_idx, msg in _progressbar(
                enumerate(all_messages),
                len(all_messages),
                prefix="  keywords",
                width=40,
            ):
                row = x_mat[row_idx]
                if row.nnz == 0:
                    continue

                indices = row.indices
                data = row.data
                top_order = data.argsort()[-TOP_KEYWORDS_PER_MESSAGE:][::-1]

                for pos in top_order:
                    score = data[pos]
                    keyword = feature_names[indices[pos]]
                    keyword_id = f"keyword::{keyword}"

                    db.upsert_node(
                        keyword_id, label="Keyword", name=keyword
                    )
                    db.add_edge_keyword(msg["id"], keyword_id, float(score))
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    print(f"  RSS after TF-IDF: {_rss_mb():.0f} MB")


def run_graph(
    pickle_path: Path,
    db_path: Path | None = None,
    export_pickle: bool = False,
    debug: bool = False,
    limit: int = 0,
    offset: int = 0,
) -> None:
    with open(pickle_path, "rb") as f:
        all_messages: list[dict[str, Any]] = pickle.load(f)

    print(f"Loaded {len(all_messages)} messages from {pickle_path}")
    print(f"  RSS: {_rss_mb():.0f} MB")

    if db_path is None:
        db_path = Path("knowledge_graph.db")

    db = GraphDB(db_path)
    processed_ids: set[str] = set()

    existing_msg_ids = {
        r["id"]
        for r in db._conn()
        .execute("SELECT id FROM node WHERE label = 'Message'")
        .fetchall()
    }
    if existing_msg_ids:
        processed_ids = existing_msg_ids
        print(f"Found existing graph at {db_path}")
        print(f"  {len(processed_ids)} messages already in graph")
        new_messages = [m for m in all_messages if m["id"] not in processed_ids]
        print(f"  {len(new_messages)} new messages to process")
    else:
        new_messages = all_messages

    if not new_messages:
        print("No new messages to process. Graph is up to date.")
        db.close()
        return

    total_new = len(new_messages)
    if offset:
        new_messages = new_messages[offset:]
        print(f"Skipped first {offset} messages ({len(new_messages)} remain)")

    if limit:
        new_messages = new_messages[:limit]
        print(f"Limited to {limit} messages (offset={offset}, total_new={total_new})")

    build_graph_to_db(new_messages, db, debug=debug, known_message_ids=processed_ids)

    print("\nDone")
    stats = db.graph_stats()
    total_nodes = stats["nodes"]["total"]
    total_edges = sum(stats["edges"].values())
    print(f"Nodes: {total_nodes}")
    print(f"Edges: {total_edges}")
    print(f"  RSS at end: {_rss_mb():.0f} MB")

    if export_pickle:
        graph_pickle_path = Path("knowledge_graph.pkl")
        graph_data = db.to_pickle()
        with open(graph_pickle_path, "wb") as f:
            pickle.dump(graph_data, f)
        print(f"Saved: {graph_pickle_path}")

    db.close()
