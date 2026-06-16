from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from pathlib import Path
    from typing import Any

from convo_tools._graph_db import GraphDB


def _conv_title(conv_id: str, db: GraphDB, edges_contains: list[tuple[str, str]]) -> str:
    msg_ids = [mid for cid, mid in edges_contains if cid == conv_id]
    for mid in msg_ids:
        n = db.get_node(mid)
        if n is not None:
            text = n.get("text")
            if isinstance(text, str) and text:
                return text[:60]
    return conv_id[:16]


def run_depth(db_path: str | Path, args: argparse.Namespace) -> None:
    import igraph as ig

    db = GraphDB(db_path)

    message_nodes = db.get_all_nodes_by_label("Message")
    edges_replies_to: list[tuple[str, str]] = db.get_edges_replies_to()
    edges_contains: list[tuple[str, str]] = db.get_edges_contains()

    if not edges_replies_to:
        print("No reply edges found in graph.")
        db.close()
        return

    id_to_idx: dict[str, int] = {}
    all_msg_ids = {n["id"] for n in message_nodes}

    def _get_idx(name: str) -> int:
        if name not in id_to_idx:
            id_to_idx[name] = len(id_to_idx)
        return id_to_idx[name]

    g_edges: list[tuple[int, int]] = []
    for p, c in edges_replies_to:
        g_edges.append((_get_idx(p), _get_idx(c)))

    for mid in all_msg_ids:
        _get_idx(mid)

    g = ig.Graph(n=len(id_to_idx), edges=g_edges, directed=True)
    g.vs["name"] = list(id_to_idx.keys())
    idx_to_id = {i: name for name, i in id_to_idx.items()}

    if not g.is_dag():
        print("Warning: reply graph has cycles. Computing depth on DAG subset.", file=sys.stderr)

    try:
        topo = g.topological_sorting()
    except Exception:
        print("Graph has cycles, computing longest paths anyway...", file=sys.stderr)
        topo = list(range(g.vcount()))

    depth: dict[str, int] = {}
    for node_id in topo:
        preds = g.predecessors(node_id)
        if not preds:
            depth[idx_to_id[node_id]] = 1
        else:
            depth[idx_to_id[node_id]] = max(depth[idx_to_id[p]] for p in preds) + 1

    conv_msg_ids: dict[str, list[str]] = defaultdict(list)
    for conv_id, msg_id in edges_contains:
        if msg_id in depth:
            conv_msg_ids[conv_id].append(msg_id)

    if not conv_msg_ids:
        print("No reply-chain messages found in any conversation.")
        db.close()
        return

    conv_metrics: list[tuple[int, str, int, float, float, str, str]] = []
    for conv_id, msg_ids in conv_msg_ids.items():
        depths = [depth[mid] for mid in msg_ids]
        max_depth = max(depths)
        mean_depth = sum(depths) / len(depths)
        n_msgs = len(msg_ids)

        children: defaultdict[str, int] = defaultdict(int)
        for p, _c in edges_replies_to:
            if p in msg_ids:
                children[p] += 1
        non_leaf = [c for c in children.values() if c > 0]
        branch_factor = sum(non_leaf) / len(non_leaf) if non_leaf else 0.0

        title = _conv_title(conv_id, db, edges_contains)
        depth_dist = " ".join(f"{d}:{depths.count(d)}" for d in sorted(set(depths)))
        conv_metrics.append((max_depth, conv_id, n_msgs, mean_depth, branch_factor, title, depth_dist))

    conv_metrics.sort(key=lambda x: (-x[0], -x[2]))

    print(f"\nConversations with reply chains: {len(conv_metrics)}")
    print()
    print(f"  {'depth':>6s}  {'msgs':>5s}  {'avg_d':>6s}  {'branch':>7s}  {'title':48s}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*48}")

    for max_depth, conv_id, n_msgs, mean_depth, branch_factor, title, _dist in conv_metrics[:args.top]:
        print(f"  {max_depth:>6d}  {n_msgs:>5d}  {mean_depth:>6.2f}  {branch_factor:>7.3f}  {title:48s}")

    all_depths = [d for cids in conv_msg_ids.values() for mid in cids for d in [depth[mid]]]
    depth_hist = Counter(all_depths)
    print(f"\nDepth distribution across all reply-chain messages ({len(all_depths)} messages):")
    max_count = max(depth_hist.values()) if depth_hist else 1
    for d in sorted(depth_hist):
        bar = "#" * int(40 * depth_hist[d] / max_count)
        print(f"  depth {d:>4d}: {bar} {depth_hist[d]}")

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["conv_id", "title", "messages_in_chain", "max_depth", "mean_depth", "branching_factor", "depth_distribution"])
            for max_depth, conv_id, n_msgs, mean_depth, branch_factor, title, dist in conv_metrics:
                w.writerow([conv_id, title, n_msgs, max_depth, f"{mean_depth:.2f}", f"{branch_factor:.3f}", dist])
        print(f"\nWrote {args.output} ({len(conv_metrics)} conversations)")

    db.close()
