import sys
import json
from pathlib import Path
import networkx as nx
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

# =====================================================
# CONFIG
# =====================================================

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

        role = (
            message.get("author", {})
            .get("role", "unknown")
        )

        content = message.get("content", {})

        if content.get("content_type") != "text":
            continue

        text = "\n".join(content.get("parts", []))

        messages.append({
            "id": node_id,
            "role": role,
            "text": text,
            "parent": node.get("parent")
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

        conversation_id = file.stem

        G.add_node(
            conversation_id,
            label="Conversation"
        )

        messages = extract_messages(conversation)

        for msg in messages:
            msg["conversation_id"] = conversation_id
            all_messages.append(msg)

            G.add_node(
                msg["id"],
                label="Message",
                role=msg["role"],
                text=msg["text"][:1000]
            )

            G.add_edge(
                conversation_id,
                msg["id"],
                relation="CONTAINS"
            )

            if msg["parent"]:
                G.add_edge(
                    msg["parent"],
                    msg["id"],
                    relation="REPLIES_TO"
                )

        print(f"  {file.name}: {len(messages)} messages")

    except Exception as e:
        print(f"ERROR {file}: {e}")

# =====================================================
# ENTITY EXTRACTION (spaCy)
# =====================================================

entity_count = 0
print(f"\nExtracting entities from {len(all_messages)} messages...")

for i, msg in enumerate(all_messages):

    if i > 0 and i % 50 == 0:
        print(f"  entities: {i}/{len(all_messages)} messages processed ({entity_count} entities found)")

    text = msg["text"]
    if not text.strip():
        continue

    try:
        doc = nlp(text)

        for ent in doc.ents:

            entity_id = f"entity::{ent.label_}::{ent.text}"

            if not G.has_node(entity_id):
                G.add_node(
                    entity_id,
                    label="Entity",
                    name=ent.text,
                    entity_type=ent.label_
                )

            G.add_edge(
                msg["id"],
                entity_id,
                relation="MENTIONS"
            )
            entity_count += 1

    except Exception as e:
        print(f"    WARN: entity extraction failed for msg {msg['id'][:20]}: {e}")

# =====================================================
# KEYWORD EXTRACTION (TF-IDF)
# =====================================================

print(f"\nExtracting TF-IDF keywords...")
corpus = [m["text"] for m in all_messages]

if corpus:

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=10000,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(corpus)

    feature_names = vectorizer.get_feature_names_out()

    for row_idx, msg in enumerate(all_messages):

        row = X[row_idx]

        scores = row.toarray()[0]

        top_indices = scores.argsort()[
            -TOP_KEYWORDS_PER_MESSAGE:
        ][::-1]

        for idx in top_indices:

            score = scores[idx]

            if score <= 0:
                continue

            keyword = feature_names[idx]

            keyword_id = f"keyword::{keyword}"

            if not G.has_node(keyword_id):

                G.add_node(
                    keyword_id,
                    label="Keyword",
                    name=keyword
                )

            G.add_edge(
                msg["id"],
                keyword_id,
                relation="HAS_KEYWORD",
                weight=float(score)
            )

# =====================================================
# CO-OCCURRENCE GRAPH BETWEEN ENTITIES
# =====================================================

cooc_edges = 0
print(f"\nBuilding entity co-occurrence edges...")

for msg in all_messages:

    entity_nodes = []

    for _, target, data in G.out_edges(
        msg["id"],
        data=True
    ):
        if data["relation"] == "MENTIONS":
            entity_nodes.append(target)

    for i in range(len(entity_nodes)):
        for j in range(i + 1, len(entity_nodes)):

            G.add_edge(
                entity_nodes[i],
                entity_nodes[j],
                relation="CO_OCCURS_WITH"
            )
            cooc_edges += 1

print(f"  Added {cooc_edges} co-occurrence edges")

# =====================================================
# EXPORT GRAPHML
# =====================================================

nx.write_graphml(
    G,
    "knowledge_graph.graphml"
)

print("\nDone")
print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())
print("Saved: knowledge_graph.graphml")

# =====================================================
# EXPORT CSV NODES
# =====================================================

nodes = []

for node, attrs in G.nodes(data=True):

    row = {"id": node}

    row.update(attrs)

    nodes.append(row)

pd.DataFrame(nodes).to_csv(
    "nodes.csv",
    index=False
)

# =====================================================
# EXPORT CSV EDGES
# =====================================================

edges = []

for source, target, attrs in G.edges(data=True):

    row = {
        "source": source,
        "target": target
    }

    row.update(attrs)

    edges.append(row)

pd.DataFrame(edges).to_csv(
    "edges.csv",
    index=False
)

print("Saved: nodes.csv")
print("Saved: edges.csv")
