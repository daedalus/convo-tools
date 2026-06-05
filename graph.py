import sys
import json
import hashlib
import pickle
from pathlib import Path
from collections import defaultdict

import networkx as nx
import pandas as pd

TOP_KEYWORDS_PER_MESSAGE = 5


def usage():
    print(
        f"Usage: {sys.argv[0]} -m <mode> [args...]",
        file=sys.stderr,
    )
    print(f"  Modes:", file=sys.stderr)
    print(f"    extract <json_dir> [pickle_path]  — read JSONs → deduped pickle", file=sys.stderr)
    print(f"    graph   <pickle_path>              — pickle → knowledge graph", file=sys.stderr)
    print(f"    full    <json_dir> [pickle_path]  — extract + graph", file=sys.stderr)
    print(f"    Default pickle_path: messages.pkl", file=sys.stderr)
    sys.exit(1)


def parse_args():
    if len(sys.argv) < 3 or sys.argv[1] != "-m":
        usage()

    mode = sys.argv[2]

    if mode in ("extract", "full"):
        if len(sys.argv) < 4:
            usage()
        json_dir = Path(sys.argv[3])
        pickle_path = Path(sys.argv[4]) if len(sys.argv) > 4 else Path("messages.pkl")
        return mode, json_dir, pickle_path

    if mode == "graph":
        pickle_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("messages.pkl")
        return mode, None, pickle_path

    usage()


def extract_messages(conversation):
    messages = []
    mapping = conversation.get("mapping", {})

    for node_id, node in mapping.items():
        message = node.get("message")
        if not message:
            continue

        role = message.get("author", {}).get("role", "unknown")
        content = message.get("content", {})

        if content.get("content_type") != "text":
            continue

        text = "\n".join(content.get("parts", []))
        messages.append({
            "id": node_id,
            "role": role,
            "text": text,
            "parent": node.get("parent"),
        })

    return messages


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_extract(json_dir: Path, pickle_path: Path):
    all_messages: list[dict] = []
    seen_hashes: set[str] = set()

    json_files = list(json_dir.glob("*.json"))
    print(f"Found {len(json_files)} conversation files in '{json_dir}/'")

    for file in json_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                conversation = json.load(f)

            conversation_id = file.stem
            messages = extract_messages(conversation)

            for msg in messages:
                msg["conversation_id"] = conversation_id
                h = text_hash(msg["text"])
                if h in seen_hashes:
                    continue
                seen_hashes.add(h)
                all_messages.append(msg)

            print(f"  {file.name}: {len(messages)} messages ({len(messages) - sum(1 for m in messages if text_hash(m['text']) in seen_hashes)} deduped)")

        except Exception as e:
            print(f"ERROR {file}: {e}")

    with open(pickle_path, "wb") as f:
        pickle.dump(all_messages, f)

    print(f"\nSaved {len(all_messages)} deduplicated messages to {pickle_path}")


def run_graph(pickle_path: Path):
    with open(pickle_path, "rb") as f:
        all_messages: list[dict] = pickle.load(f)

    print(f"Loaded {len(all_messages)} messages from {pickle_path}")

    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer

    nlp = spacy.load("en_core_web_sm")

    G = nx.MultiDiGraph()
    msg_entities: dict[str, list[str]] = defaultdict(list)

    # ── Add conversation + message nodes/edges ──
    conv_groups: dict[str, list[dict]] = defaultdict(list)
    for msg in all_messages:
        conv_groups[msg["conversation_id"]].append(msg)

    for conv_id, msgs in conv_groups.items():
        G.add_node(conv_id, label="Conversation")
        for msg in msgs:
            G.add_node(
                msg["id"],
                label="Message",
                role=msg["role"],
                text=msg["text"][:1000],
            )
            G.add_edge(conv_id, msg["id"], relation="CONTAINS")
            if msg["parent"] and G.has_node(msg["parent"]):
                G.add_edge(msg["parent"], msg["id"], relation="REPLIES_TO")

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── ENTITY EXTRACTION ──
    entity_count = 0
    print(f"\nExtracting entities from {len(all_messages)} messages...")

    max_len = nlp.max_length
    flat_texts: list[str] = []
    flat_msg_idx: list[int] = []

    for idx, msg in enumerate(all_messages):
        text = msg["text"]
        if len(text) > max_len:
            for start in range(0, len(text), max_len):
                flat_texts.append(text[start:start + max_len])
                flat_msg_idx.append(idx)
        else:
            flat_texts.append(text)
            flat_msg_idx.append(idx)

    seen_per_msg: dict[int, set[str]] = defaultdict(set)
    total_chunks = len(flat_texts)

    for i, doc in enumerate(nlp.pipe(flat_texts, batch_size=64)):
        msg_idx = flat_msg_idx[i]
        msg = all_messages[msg_idx]
        seen = seen_per_msg[msg_idx]

        if i > 0 and i % 50 == 0:
            print(f"  entities: chunk {i}/{total_chunks} ({entity_count} entities found)")

        for ent in doc.ents:
            entity_id = f"entity::{ent.label_}::{ent.text.lower()}"

            if not G.has_node(entity_id):
                G.add_node(
                    entity_id,
                    label="Entity",
                    name=ent.text.lower(),
                    entity_type=ent.label_,
                )

            if entity_id not in seen:
                G.add_edge(msg["id"], entity_id, relation="MENTIONS")
                seen.add(entity_id)
                entity_count += 1

            msg_entities[msg["id"]].append(entity_id)

    print(f"  entities: done ({entity_count} total from {total_chunks} chunks)")

    # ── KEYWORD EXTRACTION ──
    print(f"\nExtracting TF-IDF keywords...")
    corpus = [m["text"] for m in all_messages]

    if corpus:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=10000,
            ngram_range=(1, 2),
        )
        X = vectorizer.fit_transform(corpus)
        feature_names = vectorizer.get_feature_names_out()

        for row_idx, msg in enumerate(all_messages):
            row = X[row_idx]
            if row.nnz == 0:
                continue

            indices = row.indices
            data = row.data
            top_order = data.argsort()[-TOP_KEYWORDS_PER_MESSAGE:][::-1]

            for pos in top_order:
                score = data[pos]
                keyword = feature_names[indices[pos]]
                keyword_id = f"keyword::{keyword}"

                if not G.has_node(keyword_id):
                    G.add_node(keyword_id, label="Keyword", name=keyword)

                G.add_edge(
                    msg["id"],
                    keyword_id,
                    relation="HAS_KEYWORD",
                    weight=float(score),
                )

    # ── CO-OCCURRENCE ──
    cooc_edges = 0
    print(f"\nBuilding entity co-occurrence edges...")

    for msg in all_messages:
        entity_nodes = list(dict.fromkeys(msg_entities[msg["id"]]))

        for i in range(len(entity_nodes)):
            for j in range(i + 1, len(entity_nodes)):
                a, b = entity_nodes[i], entity_nodes[j]
                if a > b:
                    a, b = b, a
                G.add_edge(a, b, relation="CO_OCCURS_WITH")
                cooc_edges += 1

    print(f"  Added {cooc_edges} co-occurrence edges")

    # ── EXPORT ──
    nx.write_graphml(G, "knowledge_graph.graphml")
    print("\nDone")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Saved: knowledge_graph.graphml")

    nodes = [{"id": node, **attrs} for node, attrs in G.nodes(data=True)]
    pd.DataFrame(nodes).to_csv("nodes.csv", index=False)

    edges = [{"source": src, "target": tgt, **attrs} for src, tgt, attrs in G.edges(data=True)]
    pd.DataFrame(edges).to_csv("edges.csv", index=False)

    print("Saved: nodes.csv")
    print("Saved: edges.csv")


def main():
    mode, json_dir, pickle_path = parse_args()

    if mode in ("extract", "full"):
        run_extract(json_dir, pickle_path)

    if mode in ("graph", "full"):
        run_graph(pickle_path)


if __name__ == "__main__":
    main()
