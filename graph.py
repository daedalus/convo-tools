import sys
import json
from pathlib import Path
from collections import defaultdict

import networkx as nx
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================================
# CONFIG
# =====================================================

def usage():
    print(f"Usage: {sys.argv[0]} <data_dir>", file=sys.stderr)
    sys.exit(1)

if len(sys.argv) < 2:
    usage()

DATA_DIR = sys.argv[1]
TOP_KEYWORDS_PER_MESSAGE = 5

# =====================================================
# LOAD NLP MODEL
# =====================================================

nlp = spacy.load("en_core_web_sm")

# =====================================================
# GRAPH
# =====================================================

G = nx.MultiDiGraph()

# =====================================================
# HELPERS
# =====================================================

def extract_messages(conversation):
    """
    Extract messages from ChatGPT export format.
    """
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


# =====================================================
# COLLECT ALL MESSAGES
# =====================================================

all_messages = []

json_files = list(Path(DATA_DIR).glob("*.json"))
print(f"Found {len(json_files)} conversation files in '{DATA_DIR}/'")

for file in json_files:
    try:
        with open(file, "r", encoding="utf-8") as f:
            conversation = json.load(f)

        # Use the full path stem to reduce collision risk across subdirs
        conversation_id = file.stem

        G.add_node(conversation_id, label="Conversation")

        messages = extract_messages(conversation)

        for msg in messages:
            msg["conversation_id"] = conversation_id
            all_messages.append(msg)

            G.add_node(
                msg["id"],
                label="Message",
                role=msg["role"],
                text=msg["text"][:1000],   # stored truncated; NLP uses full text
            )

            G.add_edge(conversation_id, msg["id"], relation="CONTAINS")

            # Only wire the parent edge if the parent node actually exists
            if msg["parent"] and G.has_node(msg["parent"]):
                G.add_edge(msg["parent"], msg["id"], relation="REPLIES_TO")

        print(f"  {file.name}: {len(messages)} messages")

    except Exception as e:
        print(f"ERROR {file}: {e}")

# =====================================================
# ENTITY EXTRACTION (spaCy) — batched with nlp.pipe
# =====================================================

entity_count = 0
print(f"\nExtracting entities from {len(all_messages)} messages...")

# Precompute per-message entity lists for co-occurrence (avoids re-querying G)
msg_entities: dict[str, list[str]] = defaultdict(list)

# Flatten texts into chunks, tracking which message each belongs to
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

        # Deduplicate MENTIONS edges across chunks of the same message
        if entity_id not in seen:
            G.add_edge(msg["id"], entity_id, relation="MENTIONS")
            seen.add(entity_id)
            entity_count += 1

        msg_entities[msg["id"]].append(entity_id)

print(f"  entities: done ({entity_count} total from {total_chunks} chunks across {len(all_messages)} messages)")

# =====================================================
# KEYWORD EXTRACTION (TF-IDF)
# =====================================================

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

        # Use sparse indices/data directly — no densification
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

# =====================================================
# CO-OCCURRENCE GRAPH BETWEEN ENTITIES
# =====================================================

cooc_edges = 0
print(f"\nBuilding entity co-occurrence edges...")

# Use the precomputed dict — no G.out_edges scan per message
for msg in all_messages:
    entity_nodes = list(dict.fromkeys(msg_entities[msg["id"]]))  # deduplicated, order-preserved

    for i in range(len(entity_nodes)):
        for j in range(i + 1, len(entity_nodes)):
            a, b = entity_nodes[i], entity_nodes[j]
            # Normalize direction so (a,b) and (b,a) are the same pair
            if a > b:
                a, b = b, a
            G.add_edge(a, b, relation="CO_OCCURS_WITH")
            cooc_edges += 1

print(f"  Added {cooc_edges} co-occurrence edges")

# =====================================================
# EXPORT GRAPHML
# =====================================================

nx.write_graphml(G, "knowledge_graph.graphml")

print("\nDone")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Saved: knowledge_graph.graphml")

# =====================================================
# EXPORT CSV NODES
# =====================================================

nodes = [{"id": node, **attrs} for node, attrs in G.nodes(data=True)]
pd.DataFrame(nodes).to_csv("nodes.csv", index=False)

# =====================================================
# EXPORT CSV EDGES
# =====================================================

edges = [{"source": src, "target": tgt, **attrs} for src, tgt, attrs in G.edges(data=True)]
pd.DataFrame(edges).to_csv("edges.csv", index=False)

print("Saved: nodes.csv")
print("Saved: edges.csv")
