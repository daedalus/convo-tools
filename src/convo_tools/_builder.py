from __future__ import annotations

import gc
import itertools
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from convo_tools._util import _progressbar, _rss_mb

TOP_KEYWORDS_PER_MESSAGE = 5


def build_graph(
    all_messages: list[dict[str, Any]],
    debug: bool = False,
    limit: int = 0,
) -> dict[str, Any]:
    if limit:
        all_messages = all_messages[:limit]
        print(f"Limited to {limit} messages")

    import spacy  # noqa: PLC0415

    nlp = spacy.load(
        "en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"]
    )
    nlp.max_length = 100_000
    print(f"  RSS after spaCy load: {_rss_mb():.0f} MB")

    nodes: dict[str, dict[str, Any]] = {}
    edges_contains: set[tuple[str, str]] = set()
    edges_replies_to: set[tuple[str, str]] = set()
    edges_mentions: set[tuple[str, str]] = set()
    edges_cooc: set[tuple[str, str]] = set()
    edges_keywords: list[tuple[str, str, float]] = []
    gc.collect()

    # ── Add conversation + message nodes/edges ──
    conv_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for msg in all_messages:
        conv_groups[msg["conversation_id"]].append(msg)

    for conv_id, msgs in conv_groups.items():
        nodes[conv_id] = {"label": "Conversation"}
        for msg in msgs:
            nodes[msg["id"]] = {
                "label": "Message",
                "role": msg["role"],
                "text": msg["text"][:1000],
            }
            edges_contains.add((conv_id, msg["id"]))
            if msg["parent"] and msg["parent"] in nodes:
                edges_replies_to.add((msg["parent"], msg["id"]))

    total_edges = len(edges_contains) + len(edges_replies_to)
    print(f"Graph: {len(nodes)} nodes, {total_edges} edges")
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
                    if entity_id not in nodes:
                        nodes[entity_id] = {
                            "label": "Entity",
                            "name": ent.text.lower(),
                            "entity_type": ent.label_,
                        }
                    if entity_id not in seen_per_msg[offset]:
                        edges_mentions.add((msg["id"], entity_id))
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
                if (a, b) not in edges_cooc:
                    edges_cooc.add((a, b))
                    cooc_edges += 1

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

        for row_idx, msg in _progressbar(
            enumerate(all_messages), len(all_messages), prefix="  keywords", width=40
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

                if keyword_id not in nodes:
                    nodes[keyword_id] = {"label": "Keyword", "name": keyword}

                edges_keywords.append((msg["id"], keyword_id, float(score)))

    print(f"  RSS after TF-IDF: {_rss_mb():.0f} MB")

    return {
        "nodes": nodes,
        "edges_contains": edges_contains,
        "edges_replies_to": edges_replies_to,
        "edges_mentions": edges_mentions,
        "edges_cooc": edges_cooc,
        "edges_keywords": edges_keywords,
    }


def run_graph(
    pickle_path: Path,
    export_pickle: bool = False,
    debug: bool = False,
    limit: int = 0,
    offset: int = 0,
) -> None:
    with open(pickle_path, "rb") as f:
        all_messages: list[dict[str, Any]] = pickle.load(f)

    print(f"Loaded {len(all_messages)} messages from {pickle_path}")
    print(f"  RSS: {_rss_mb():.0f} MB")

    graph_pickle_path = Path("knowledge_graph.pkl")
    existing_graph: dict[str, Any] | None = None
    processed_ids: set[str] = set()

    if graph_pickle_path.exists():
        print(f"Found existing graph at {graph_pickle_path}, loading...")
        with open(graph_pickle_path, "rb") as f:
            loaded = pickle.load(f)
        if isinstance(loaded, dict):
            existing_graph = loaded
            processed_ids = existing_graph.get("processed_message_ids", set())
            print(f"  {len(processed_ids)} messages already in graph")
        else:
            print(
                f"  Unrecognized format ({type(loaded).__name__}), "
                "rebuilding from scratch."
            )
        new_messages = [m for m in all_messages if m["id"] not in processed_ids]
        print(f"  {len(new_messages)} new messages to process")
    else:
        new_messages = all_messages

    if not new_messages:
        print("No new messages to process. Graph is up to date.")
        return

    total_new = len(new_messages)
    if offset:
        new_messages = new_messages[offset:]
        print(f"Skipped first {offset} messages ({len(new_messages)} remain)")

    if limit:
        new_messages = new_messages[:limit]
        print(f"Limited to {limit} messages (offset={offset}, total_new={total_new})")

    graph_data = build_graph(new_messages, debug=debug, limit=0)

    if existing_graph:
        graph_data["nodes"] = {**existing_graph["nodes"], **graph_data["nodes"]}
        graph_data["edges_contains"] = existing_graph["edges_contains"] | graph_data["edges_contains"]
        graph_data["edges_replies_to"] = existing_graph["edges_replies_to"] | graph_data["edges_replies_to"]
        graph_data["edges_mentions"] = existing_graph["edges_mentions"] | graph_data["edges_mentions"]
        graph_data["edges_cooc"] = existing_graph["edges_cooc"] | graph_data["edges_cooc"]
        graph_data["edges_keywords"] = existing_graph["edges_keywords"] + graph_data["edges_keywords"]
        processed_ids |= {m["id"] for m in new_messages}
    else:
        processed_ids = {m["id"] for m in new_messages}

    graph_data["processed_message_ids"] = processed_ids

    print("\nDone")
    total_edges = (
        len(graph_data["edges_contains"])
        + len(graph_data["edges_replies_to"])
        + len(graph_data["edges_mentions"])
        + len(graph_data["edges_cooc"])
        + len(graph_data["edges_keywords"])
    )
    print("Nodes:", len(graph_data["nodes"]))
    print("Edges:", total_edges)
    print(f"  RSS at end: {_rss_mb():.0f} MB")

    if export_pickle:
        with open(graph_pickle_path, "wb") as f:
            pickle.dump(graph_data, f)
        print(f"Saved: {graph_pickle_path}")
