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


def _prune_graph(g, max_edges_per_node: int = 100) -> None:
    """KNN prune: keep only the strongest edges per node."""
    if not max_edges_per_node:
        print(f"  Graph: {g.vcount()} nodes, {g.ecount()} edges")
        return

    avg_degree = g.ecount() * 2 / g.vcount() if g.vcount() else 0
    if avg_degree <= max_edges_per_node:
        print(f"  Graph: {g.vcount()} nodes, {g.ecount()} edges (avg degree {avg_degree:.0f})")
        return

    print(f"  Pruning from avg degree {avg_degree:.0f} to {max_edges_per_node}...")
    edges_to_remove = []
    for v in g.vs:
        nbs = g.neighbors(v.index)
        if len(nbs) > max_edges_per_node:
            edge_weights = [(nb, g[v.index, nb]) for nb in nbs]
            edge_weights.sort(key=lambda x: -x[1])
            for nb, _ in edge_weights[max_edges_per_node:]:
                edges_to_remove.append(g.get_eid(v.index, nb))
    g.delete_edges(edges_to_remove)
    print(f"  Pruned to {g.vcount()} nodes, {g.ecount()} edges (avg degree {g.ecount() * 2 / g.vcount():.0f})")


def compute_node2vec_embeddings(
    db: GraphDB,
    dim: int = 32,
    walk_length: int = 40,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    min_edge_weight: int = 2,
    max_edges_per_node: int = 100,
    include_keywords: bool = False,
) -> dict[str, np.ndarray]:
    id_to_idx: dict[str, int] = {}
    edge_pairs: list[tuple[int, int]] = []

    def _get_idx(name: str) -> int:
        if name not in id_to_idx:
            id_to_idx[name] = len(id_to_idx)
        return id_to_idx[name]

    for src, dst in db.get_edges_mentions():
        edge_pairs.append((_get_idx(src), _get_idx(dst)))

    for r in db._conn().execute(
        "SELECT a.entity_id AS entity_a, b.entity_id AS entity_b, weight "
        "FROM edge_cooc "
        "JOIN entity_int a ON edge_cooc.entity_a_int = a.int_id "
        "JOIN entity_int b ON edge_cooc.entity_b_int = b.int_id "
        "WHERE weight >= ?",
        (min_edge_weight,),
    ):
        edge_pairs.append((_get_idx(r["entity_a"]), _get_idx(r["entity_b"])))

    if include_keywords:
        for src, dst, _w in db.get_edges_keywords():
            edge_pairs.append((_get_idx(src), _get_idx(dst)))

    n_nodes = len(id_to_idx)
    if n_nodes < 3:
        return {}

    g = ig.Graph(n=n_nodes, edges=edge_pairs, directed=False)
    del edge_pairs
    g.simplify(multiple=True, loops=True)

    _prune_graph(g, max_edges_per_node)

    import random
    random.seed(42)

    idx_to_id = {i: name for name, i in id_to_idx.items()}

    import array as _array
    from scipy.sparse import coo_matrix, csr_matrix
    import numpy as np

    cooc = csr_matrix((n_nodes, n_nodes), dtype=np.float32)
    window = 5

    total_walks = num_walks * n_nodes
    walk_count = 0
    BATCH = 50000

    import time as _time
    _t0 = _time.monotonic()

    node_indices = list(range(n_nodes))
    for _ in range(num_walks):
        random.shuffle(node_indices)
        batch_rows = _array.array("l")
        batch_cols = _array.array("l")

        for node in node_indices:
            walk = [node]
            for _ in range(walk_length - 1):
                current = walk[-1]
                nbs = g.neighbors(current)
                if not nbs:
                    break
                if len(walk) >= 2:
                    prev = walk[-2]
                    prev_nbs = set(g.neighbors(prev))
                    biased_nbs = []
                    for nb in nbs:
                        if nb == prev:
                            biased_nbs.extend([nb] * int(1.0 / p))
                        elif nb in prev_nbs:
                            biased_nbs.extend([nb] * int(1.0 / q))
                        else:
                            biased_nbs.append(nb)
                    if biased_nbs:
                        walk.append(random.choice(biased_nbs))
                    else:
                        walk.append(random.choice(nbs))
                else:
                    walk.append(random.choice(nbs))

            for i, node in enumerate(walk):
                for j in range(max(0, i - window), min(len(walk), i + window + 1)):
                    if i != j:
                        batch_rows.append(node)
                        batch_cols.append(walk[j])

            walk_count += 1
            if walk_count % BATCH == 0:
                data = np.ones(len(batch_rows), dtype=np.float32)
                cooc = cooc + coo_matrix((data, (list(batch_rows), list(batch_cols))), shape=(n_nodes, n_nodes)).tocsr()
                del batch_rows, batch_cols, data
                batch_rows = _array.array("l")
                batch_cols = _array.array("l")
                elapsed = _time.monotonic() - _t0
                rate = walk_count / elapsed if elapsed > 0 else 0
                eta = (total_walks - walk_count) / rate if rate > 0 else 0
                if eta >= 3600:
                    eta_str = f"{eta / 3600:.1f}h"
                elif eta >= 60:
                    eta_str = f"{eta / 60:.1f}m"
                else:
                    eta_str = f"{eta:.0f}s"
                print(f"\r  walks {walk_count}/{total_walks}  {rate:.0f}/s  ETA {eta_str}  ", end="", flush=True)

        if batch_rows:
            data = np.ones(len(batch_rows), dtype=np.float32)
            cooc = cooc + coo_matrix((data, (list(batch_rows), list(batch_cols))), shape=(n_nodes, n_nodes)).tocsr()
            del batch_rows, batch_cols, data

    print(f"\r  {walk_count} walks completed              ")

    cooc_csr = cooc
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
