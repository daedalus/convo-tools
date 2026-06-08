import sys
import json
import gc
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
    print(f"    extract <json_dir> [pickle_path]   — read JSONs → deduped pickle", file=sys.stderr)
    print(f"    graph   <pickle_path> [--pickle] [--debug] [--limit N]  — pickle → knowledge graph", file=sys.stderr)
    print(f"    full    <json_dir> [pickle_path]              — extract + graph", file=sys.stderr)
    print(f"    Default pickle_path: messages.pkl", file=sys.stderr)
    print(f"    --pickle  also export G as knowledge_graph.pkl", file=sys.stderr)
    print(f"    --debug   print each message and its extracted entities", file=sys.stderr)
    print(f"    --limit N only process first N messages", file=sys.stderr)
    sys.exit(1)


def parse_args():
    if len(sys.argv) < 3 or sys.argv[1] != "-m":
        usage()

    mode = sys.argv[2]

    remaining = sys.argv[3:]
    pickle_path = Path("messages.pkl")
    json_dir = None
    export_pickle = False
    debug = False
    limit = 0

    for arg in remaining:
        if arg == "--pickle":
            export_pickle = True
        elif arg == "--debug":
            debug = True
        elif arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])
        elif arg.startswith("--"):
            continue
        elif json_dir is None and mode in ("extract", "full"):
            json_dir = Path(arg)
        elif pickle_path == Path("messages.pkl"):
            pickle_path = Path(arg)
        else:
            usage()

    if mode in ("extract", "full") and json_dir is None:
        usage()

    return mode, json_dir, pickle_path, export_pickle, debug, limit


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


def run_graph(pickle_path: Path, export_pickle: bool = False, debug: bool = False, limit: int = 0):
    with open(pickle_path, "rb") as f:
        all_messages: list[dict] = pickle.load(f)

    print(f"Loaded {len(all_messages)} messages from {pickle_path}")

    if limit:
        all_messages = all_messages[:limit]
        print(f"Limited to {limit} messages")

    import spacy
    from sklearn.feature_extraction.text import TfidfVectorizer

    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    nlp.max_length = 100_000

    G = nx.MultiDiGraph()

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

    cooc_edges = 0
    next_print = 500

    for batch_start in range(0, len(all_messages), 16):
        batch = all_messages[batch_start:batch_start + 16]

        seen_per_msg: dict[int, set[str]] = defaultdict(set)

        for offset, msg in enumerate(batch):
            text = msg["text"]
            if len(text) > max_len:
                chunks = [text[s:s + max_len] for s in range(0, len(text), max_len)]
            else:
                chunks = [text]

            msg_entities_list: list[str] = []

            for chunk in chunks:
                doc = nlp(chunk)
                for ent in doc.ents:
                    entity_id = f"entity::{ent.label_}::{ent.text.lower()}"
                    if not G.has_node(entity_id):
                        G.add_node(entity_id, label="Entity", name=ent.text.lower(), entity_type=ent.label_)
                    if entity_id not in seen_per_msg[offset]:
                        G.add_edge(msg["id"], entity_id, relation="MENTIONS")
                        seen_per_msg[offset].add(entity_id)
                        entity_count += 1
                        msg_entities_list.append(f"{ent.label_}:{ent.text}")

            if debug and msg_entities_list:
                print(f"  msg {msg['id']}: {', '.join(msg_entities_list)}")

        if entity_count >= next_print:
            print(f"  entities: {entity_count} found...")
            next_print = entity_count + 500

        gc.collect()

        for entities in seen_per_msg.values():
            entity_list = list(entities)
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    a, b = entity_list[i], entity_list[j]
                    if a > b:
                        a, b = b, a
                    G.add_edge(a, b, relation="CO_OCCURS_WITH")
                    cooc_edges += 1

    print(f"  entities: done ({entity_count} total)")
    print(f"  Added {cooc_edges} co-occurrence edges")
    gc.collect()

    # ── KEYWORD EXTRACTION ──
    print(f"\nExtracting TF-IDF keywords...")
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2),
    )
    X = vectorizer.fit_transform(m["text"] for m in all_messages)
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

    if export_pickle:
        with open("knowledge_graph.pkl", "wb") as f:
            pickle.dump(G, f)
        print("Saved: knowledge_graph.pkl")


def main():
    mode, json_dir, pickle_path, export_pickle, debug, limit = parse_args()

    if mode in ("extract", "full"):
        run_extract(json_dir, pickle_path)

    if mode in ("graph", "full"):
        run_graph(pickle_path, export_pickle, debug, limit)


if __name__ == "__main__":
    main()
