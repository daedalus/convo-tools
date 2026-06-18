from __future__ import annotations

import csv
import pickle
import sys
from typing import TYPE_CHECKING

from convo_tools._util import safe_pickle_load

if TYPE_CHECKING:
    import argparse
    from typing import Any

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def _msg_text(nodes: dict[str, Any], msg_id: str) -> str:
    n = nodes.get(msg_id, {})
    if isinstance(n, dict):
        t = n.get("text", "")
        return str(t)[:80] if t else ""
    return ""


def _msg_role(nodes: dict[str, Any], msg_id: str) -> str:
    n = nodes.get(msg_id, {})
    if isinstance(n, dict):
        r = n.get("role", "")
        return str(r) if r else "?"
    return "?"


def _load_graph(pickle_path: str) -> dict[str, Any] | None:
    graph = safe_pickle_load(pickle_path)
    if not isinstance(graph, dict) or "nodes" not in graph:
        print("Error: graph pickle missing 'nodes' key", file=sys.stderr)
        return None
    return graph


def _compute_embeddings(
    graph: dict[str, Any],
    dim: int,
    n_iter: int,
    include_keywords: bool,
) -> tuple[np.ndarray, list[str], dict[str, Any]] | None:
    nodes = graph.get("nodes", {})
    edges_mentions = graph.get("edges_mentions", set())
    edges_keywords = graph.get("edges_keywords", [])

    message_ids = sorted(
        nid for nid, attrs in nodes.items() if isinstance(attrs, dict) and attrs.get("label") == "Message"
    )
    entity_ids = sorted(
        nid for nid, attrs in nodes.items()
        if isinstance(attrs, dict) and attrs.get("label") not in ("Conversation", "Message", "Keyword")
    )
    keyword_ids = sorted(
        nid for nid, attrs in nodes.items() if isinstance(attrs, dict) and attrs.get("label") == "Keyword"
    )

    n_msgs = len(message_ids)
    n_ents = len(entity_ids)
    n_kws = len(keyword_ids)

    if n_msgs == 0:
        print("  No message nodes found in graph.")
        return None

    msg_index: dict[str, int] = {mid: i for i, mid in enumerate(message_ids)}

    #── Build entity incidence ──
    ent_index: dict[str, int] = {eid: i for i, eid in enumerate(entity_ids)}
    ent_rows: list[int] = []
    ent_cols: list[int] = []
    ent_vals: list[float] = []

    for msg_id, ent_id in edges_mentions:
        mi = msg_index.get(msg_id)
        ei = ent_index.get(ent_id)
        if mi is not None and ei is not None:
            ent_rows.append(mi)
            ent_cols.append(ei)
            ent_vals.append(1.0)

    m_ent = csr_matrix(
        (ent_vals, (ent_rows, ent_cols)),
        shape=(n_msgs, max(n_ents, 1)),
        dtype=np.float64,
    )

    if include_keywords and n_kws > 0:
        kw_index: dict[str, int] = {kid: i for i, kid in enumerate(keyword_ids)}
        kw_rows: list[int] = []
        kw_cols: list[int] = []
        kw_vals: list[float] = []
        for msg_id, kid, _score in edges_keywords:
            mi = msg_index.get(msg_id)
            ki = kw_index.get(kid)
            if mi is not None and ki is not None:
                kw_rows.append(mi)
                kw_cols.append(ki)
                kw_vals.append(1.0)
        inc_mat = csr_matrix(
            (ent_vals + kw_vals, (ent_rows + kw_rows, ent_cols + [n_ents + c for c in kw_cols])),
            shape=(n_msgs, n_ents + n_kws),
            dtype=np.float64,
        )
    else:
        inc_mat = m_ent

    nnz = inc_mat.nnz
    print(f"  Messages: {n_msgs}")
    col_desc = f"Entity columns: {n_ents}"
    if include_keywords and n_kws > 0:
        col_desc += f" + {n_kws} keywords"
    print(f"  {col_desc}")
    print(f"  Incidence non-zeros: {nnz}")
    print(f"  Density: {100.0 * nnz / (n_msgs * inc_mat.shape[1]):.4f}%")
    print()

    max_dim = min(inc_mat.shape) - 1
    if dim <= 0 or dim > max_dim:
        dim = max_dim
    if dim < 1:
        print("  Incidence matrix too small for embedding (need at least 2 rows/cols).")
        return None

    print(f"  Computing TruncatedSVD (d={dim})...")
    svd = TruncatedSVD(n_components=dim, random_state=42, n_iter=n_iter)
    emb = svd.fit_transform(inc_mat)
    emb = normalize(emb, norm="l2")
    var_explained = svd.explained_variance_ratio_.sum()
    print(f"  Variance explained: {var_explained:.3f}")
    print(f"  Embedding shape: {emb.shape}")
    print()

    return emb, message_ids, nodes


def _find_similar(
    emb: np.ndarray,
    message_ids: list[str],
    nodes: dict[str, Any],
    query_id: str,
    top_k: int,
) -> list[tuple[str, float, str, str]]:
    msg_index = {mid: i for i, mid in enumerate(message_ids)}
    qi = msg_index.get(query_id)
    if qi is None:
        print(f"  Error: message ID not found: {query_id}", file=sys.stderr)
        return []

    n_msgs = emb.shape[0]
    nn = NearestNeighbors(n_neighbors=min(top_k + 1, n_msgs), metric="cosine")
    nn.fit(emb)
    distances, indices = nn.kneighbors(emb[qi: qi + 1])

    print(f"  Query: {_msg_role(nodes, query_id):>8s} | {_msg_text(nodes, query_id)}")
    print(f"  Top {top_k} similar messages:")
    print(f"  {'':>4s}  {'score':>6s}  {'role':>8s}  {'text':60s}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*8}  {'─'*60}")

    results: list[tuple[str, float, str, str]] = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
        if idx == qi:
            continue
        sim = float(1.0 - dist)
        mid = message_ids[idx]
        role = _msg_role(nodes, mid)
        text = _msg_text(nodes, mid)
        results.append((mid, sim, role, text))
        print(f"  {rank:>4d}  {sim:>6.4f}  {role:>8s}  {text:60s}")
        if rank >= top_k:
            break

    return results


def run_embed(args: argparse.Namespace) -> None:
    #── Load precomputed embeddings ──
    if args.load:
        loaded = np.load(args.load)
        emb = loaded["embedding"]
        message_ids = [str(mid) for mid in loaded["message_ids"]]
        print(f"  Loaded embeddings from {args.load}: {emb.shape}")

        graph = _load_graph(str(args.pickle_path))
        nodes = graph.get("nodes", {}) if graph else {}

        if args.similar_to:
            results = _find_similar(emb, message_ids, nodes, args.similar_to, args.top)
            if args.output and results:
                with open(args.output, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow(["neighbor_msg_id", "cosine_similarity", "role", "text_preview"])
                    for mid, sim, role, text in results:
                        w.writerow([mid, f"{sim:.4f}", role, text])
                print(f"\n  Wrote {args.output} ({len(results)} neighbors)")
        return

    #── Compute from scratch ──
    graph = _load_graph(str(args.pickle_path))
    if graph is None:
        return

    result = _compute_embeddings(graph, args.dim, args.n_iter, args.all)
    if result is None:
        return

    emb, message_ids, nodes = result

    if args.save:
        np.savez_compressed(
            args.save,
            embedding=emb,
            message_ids=message_ids,
        )
        print(f"  Saved embeddings to {args.save}")

    if args.similar_to:
        results = _find_similar(emb, message_ids, nodes, args.similar_to, args.top)
        if args.output and results:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["neighbor_msg_id", "cosine_similarity", "role", "text_preview"])
                for mid, sim, role, text in results:
                    w.writerow([mid, f"{sim:.4f}", role, text])
            print(f"\n  Wrote {args.output} ({len(results)} neighbors)")
    elif args.output:
        print("  --output requires --similar-to (otherwise there's no query to export)")
