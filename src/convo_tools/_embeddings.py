from __future__ import annotations

import gc
import json
import sqlite3
from typing import TYPE_CHECKING, Any

import igraph as ig
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

from convo_tools._graph_db import GraphDB

EMBEDDING_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS node_embedding (
    node_id TEXT PRIMARY KEY,
    embedding TEXT NOT NULL,
    dim INTEGER NOT NULL,
    FOREIGN KEY (node_id) REFERENCES node(id)
);

CREATE INDEX IF NOT EXISTS idx_emb_dim ON node_embedding(dim);
"""


def _init_embedding_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(EMBEDDING_SCHEMA_SQL)
    conn.commit()


def _build_sparse_incidence(
    db: GraphDB,
    include_keywords: bool = False,
) -> tuple[list[str], dict[str, int], Any]:
    from scipy.sparse import lil_matrix

    msg_entities = db.get_message_entities()
    msg_ids = sorted(msg_entities.keys())
    if not msg_ids:
        return [], {}, None

    all_features: set[str] = set()
    for mid in msg_ids:
        all_features.update(msg_entities[mid])
    if include_keywords:
        msg_keywords = db.get_message_keywords()
        for mid in msg_ids:
            all_features.update(msg_keywords.get(mid, set()))

    feature_list = sorted(all_features)
    feature_idx = {f: i for i, f in enumerate(feature_list)}

    matrix = lil_matrix((len(msg_ids), len(feature_list)), dtype=np.float32)
    for row, mid in enumerate(msg_ids):
        for feat in msg_entities[mid]:
            matrix[row, feature_idx[feat]] = 1.0
        if include_keywords:
            for feat in msg_keywords.get(mid, set()):
                matrix[row, feature_idx[feat]] = 1.0

    return msg_ids, feature_idx, matrix.tocsr()


def compute_spectral_embeddings(
    db: GraphDB,
    dim: int = 32,
    include_keywords: bool = False,
    n_iter: int = 5,
) -> dict[str, np.ndarray]:
    msg_ids, feature_idx, matrix = _build_sparse_incidence(db, include_keywords)
    if matrix is None or matrix.shape[0] < 2:
        return {}

    k = min(dim, min(matrix.shape) - 1, 64)
    if k < 1:
        return {}

    print(f"  Matrix shape: {matrix.shape} (sparse, {matrix.nnz} non-zero)")

    try:
        from scipy.sparse.linalg import svds

        print(f"  Running sparse SVD with k={k}...")
        U, _, _ = svds(matrix.astype(np.float64), k=k, niter=n_iter)
    except Exception:
        from numpy.linalg import svd

        print("  Falling back to dense SVD...")
        U, _, _ = svd(matrix.toarray().astype(np.float64), full_matrices=False)
        U = U[:, :k]

    del matrix
    gc.collect()

    embeddings = {}
    for i, mid in enumerate(msg_ids):
        embeddings[mid] = U[i].astype(np.float32)

    return embeddings


def compute_node2vec_embeddings(
    db: GraphDB,
    dim: int = 32,
    walk_length: int = 40,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    include_keywords: bool = False,
) -> dict[str, np.ndarray]:
    id_to_idx: dict[str, int] = {}
    edges: list[tuple[int, int]] = []
    weights: list[float] = []

    def _get_idx(name: str) -> int:
        if name not in id_to_idx:
            id_to_idx[name] = len(id_to_idx)
        return id_to_idx[name]

    for src, dst in db.get_edges_mentions():
        edges.append((_get_idx(src), _get_idx(dst)))
        weights.append(1.0)

    for r in db._conn().execute(
        "SELECT a.entity_id AS entity_a, b.entity_id AS entity_b, weight "
        "FROM edge_cooc "
        "JOIN entity_int a ON edge_cooc.entity_a_int = a.int_id "
        "JOIN entity_int b ON edge_cooc.entity_b_int = b.int_id"
    ):
        edges.append((_get_idx(r["entity_a"]), _get_idx(r["entity_b"])))
        weights.append(float(r["weight"]))

    if include_keywords:
        for r in db.get_edges_keywords():
            src, dst, weight = r["msg_id"], r["keyword_id"], r["weight"]
            edges.append((_get_idx(src), _get_idx(dst)))
            weights.append(weight)

    n_nodes = len(id_to_idx)
    if n_nodes < 3:
        return {}

    g = ig.Graph(n=n_nodes, edges=edges, directed=False)
    g.simplify(multiple=True, loops=True, combine_edges="sum")

    print(f"  Graph: {g.vcount()} nodes, {g.ecount()} edges")

    from collections import defaultdict

    neighbors: dict[int, list[int]] = defaultdict(list)
    for e in g.es:
        neighbors[e.source].append(e.target)
        neighbors[e.target].append(e.source)

    import random
    random.seed(42)

    walks: list[list[str]] = []
    idx_to_id = {i: name for name, i in id_to_idx.items()}
    node_indices = list(range(n_nodes))
    for _ in range(num_walks):
        random.shuffle(node_indices)
        for node in node_indices:
            walk = [node]
            for _ in range(walk_length - 1):
                current = walk[-1]
                nbs = neighbors.get(current, [])
                if not nbs:
                    break
                if len(walk) >= 2:
                    prev = walk[-2]
                    biased_nbs = []
                    for nb in nbs:
                        if nb == prev:
                            biased_nbs.extend([nb] * int(1.0 / p))
                        elif nb in neighbors.get(prev, []):
                            biased_nbs.extend([nb] * int(1.0 / q))
                        else:
                            biased_nbs.append(nb)
                    if biased_nbs:
                        walk.append(random.choice(biased_nbs))
                    else:
                        walk.append(random.choice(nbs))
                else:
                    walk.append(random.choice(nbs))
            walks.append([idx_to_id[i] for i in walk])

    print(f"  {len(walks)} walks completed")

    vocab = id_to_idx
    n_nodes = len(vocab)

    from scipy.sparse import lil_matrix

    cooc = lil_matrix((n_nodes, n_nodes), dtype=np.float32)
    window = 5
    for walk in walks:
        for i, node in enumerate(walk):
            vi = vocab[node]
            for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                if i != j:
                    cooc[vi, vocab[walk[j]]] += 1.0

    del walks
    gc.collect()

    cooc_csr = cooc.tocsr()
    del cooc
    gc.collect()

    from scipy.sparse.linalg import svds

    k = min(dim, min(cooc_csr.shape) - 1, 64)
    if k < 1:
        return {}

    try:
        U, _, _ = svds(cooc_csr.astype(np.float64), k=k, niter=5)
    except Exception:
        from numpy.linalg import svd
        U, _, _ = svd(cooc_csr.toarray().astype(np.float64), full_matrices=False)
        U = U[:, :k]

    del cooc_csr
    gc.collect()

    embeddings = {}
    for node, idx in vocab.items():
        embeddings[node] = U[idx].astype(np.float32)

    return embeddings


def store_embeddings(
    db: GraphDB,
    embeddings: dict[str, np.ndarray],
    batch_size: int = 1000,
) -> int:
    conn = db._conn()
    _init_embedding_schema(conn)

    count = 0
    batch = []
    for node_id, vec in embeddings.items():
        batch.append((node_id, json.dumps(vec.tolist()), len(vec)))
        if len(batch) >= batch_size:
            conn.executemany(
                "INSERT OR REPLACE INTO node_embedding (node_id, embedding, dim) "
                "VALUES (?, ?, ?)",
                batch,
            )
            count += len(batch)
            batch = []

    if batch:
        conn.executemany(
            "INSERT OR REPLACE INTO node_embedding (node_id, embedding, dim) "
            "VALUES (?, ?, ?)",
            batch,
        )
        count += len(batch)

    conn.commit()
    return count


def load_embeddings(db: GraphDB) -> dict[str, np.ndarray]:
    conn = db._conn()
    _init_embedding_schema(conn)

    rows = conn.execute("SELECT node_id, embedding FROM node_embedding").fetchall()
    embeddings = {}
    for r in rows:
        embeddings[r["node_id"]] = np.array(json.loads(r["embedding"]), dtype=np.float32)
    return embeddings


def find_similar_nodes(
    db: GraphDB,
    query_id: str,
    top: int = 10,
    min_similarity: float = 0.0,
) -> list[tuple[str, float]]:
    embeddings = load_embeddings(db)
    if query_id not in embeddings:
        return []

    query_vec = embeddings[query_id]
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []

    scores: list[tuple[str, float]] = []
    for node_id, vec in embeddings.items():
        if node_id == query_id:
            continue
        sim = float(np.dot(query_vec, vec) / (query_norm * np.linalg.norm(vec)))
        if sim >= min_similarity:
            scores.append((node_id, sim))

    scores.sort(key=lambda x: -x[1])
    return scores[:top]


def find_similar_conversations(
    db: GraphDB,
    conv_id: str,
    top: int = 10,
    min_similarity: float = 0.0,
) -> list[tuple[str, float]]:
    embeddings = load_embeddings(db)
    if conv_id not in embeddings:
        return []

    conv_vec = embeddings[conv_id]
    conv_norm = np.linalg.norm(conv_vec)
    if conv_norm == 0:
        return []

    conv_ids = {
        r["id"] for r in db.get_all_nodes_by_label("Conversation")
    }

    scores: list[tuple[str, float]] = []
    for cid in conv_ids:
        if cid == conv_id or cid not in embeddings:
            continue
        vec = embeddings[cid]
        sim = float(np.dot(conv_vec, vec) / (conv_norm * np.linalg.norm(vec)))
        if sim >= min_similarity:
            scores.append((cid, sim))

    scores.sort(key=lambda x: -x[1])
    return scores[:top]


def run_embeddings(
    db: GraphDB,
    dim: int = 32,
    method: str = "spectral",
    include_keywords: bool = False,
) -> dict[str, int]:
    if method == "spectral":
        print("Computing spectral embeddings...")
        msg_embs = compute_spectral_embeddings(db, dim=dim, include_keywords=include_keywords)
        print(f"  {len(msg_embs)} message embeddings")
    elif method == "node2vec":
        print("Computing node2vec embeddings...")
        msg_embs = compute_node2vec_embeddings(db, dim=dim, include_keywords=include_keywords)
        print(f"  {len(msg_embs)} node embeddings")
    else:
        raise ValueError(f"Unknown method: {method}")

    if not msg_embs:
        print("  No embeddings computed")
        return {"message_embeddings": 0, "conversation_embeddings": 0, "total_stored": 0}

    conv_ids = [r["id"] for r in db.get_all_nodes_by_label("Conversation")]
    conv_msgs = db.get_conv_msgs_map()

    conv_embs: dict[str, np.ndarray] = {}
    for cid in conv_ids:
        member_ids = conv_msgs.get(cid, [])
        member_embs = [msg_embs[mid] for mid in member_ids if mid in msg_embs]
        if member_embs:
            conv_embs[cid] = np.mean(member_embs, axis=0).astype(np.float32)

    all_embs = {**msg_embs, **conv_embs}

    n = store_embeddings(db, all_embs)
    print(f"  Stored {n} embeddings")

    return {
        "message_embeddings": len(msg_embs),
        "conversation_embeddings": len(conv_embs),
        "total_stored": n,
    }
