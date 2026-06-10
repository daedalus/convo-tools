from __future__ import annotations

import csv
import pickle
import sys
from collections import Counter, defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from typing import Any

import networkx as nx


def _conv_title(conv_id: str, nodes: dict[str, Any], edges_contains: set[tuple[str, str]]) -> str:
    msg_ids = [mid for cid, mid in edges_contains if cid == conv_id]
    for mid in msg_ids:
        n = nodes.get(mid, {})
        if isinstance(n, dict):
            text = n.get("text")
            if isinstance(text, str) and text:
                return text[:60]
    return conv_id[:16]


def run_depth(args: argparse.Namespace) -> None:
    with open(args.pickle_path, "rb") as f:
        graph = pickle.load(f)

    if not isinstance(graph, dict) or "nodes" not in graph:
        print("Error: graph pickle missing 'nodes' key", file=sys.stderr)
        return

    nodes = graph.get("nodes", {})
    edges_replies_to = graph.get("edges_replies_to", set())
    edges_contains = graph.get("edges_contains", set())

    if not edges_replies_to:
        print("No reply edges found in graph.")
        return

    g = nx.DiGraph()
    g.add_edges_from(edges_replies_to)
    all_msg_ids = {nid for nid, attrs in nodes.items() if attrs.get("label") == "Message"}
    g.add_nodes_from(all_msg_ids)

    if not nx.is_directed_acyclic_graph(g):
        cycles = list(nx.simple_cycles(g))
        print(f"Warning: reply graph has {len(cycles)} cycle(s). Computing DAG on largest acyclic subset.", file=sys.stderr)
        for cycle in cycles:
            print(f"  Cycle: {' -> '.join(cycle[:5])}{'...' if len(cycle) > 5 else ''}", file=sys.stderr)

    # Depth via topological DP (longest path in DAG)
    try:
        topo = list(nx.topological_sort(g))
    except nx.NetworkXUnfeasible:
        print("Graph has cycles, removing back edges...")
        g = nx.DiGraph(nx.algorithms.dag.transitive_reduction(nx.DiGraph(g) if nx.is_directed_acyclic_graph(g) else nx.DiGraph()))
        topo = list(nx.topological_sort(g))

    depth: dict[str, int] = {}
    for node_id in topo:
        preds = list(g.predecessors(node_id))
        if not preds:
            depth[node_id] = 1
        else:
            depth[node_id] = max(depth[p] for p in preds) + 1

    # Group messages by conversation
    conv_msg_ids: dict[str, list[str]] = defaultdict(list)
    for conv_id, msg_id in edges_contains:
        if msg_id in depth:
            conv_msg_ids[conv_id].append(msg_id)

    if not conv_msg_ids:
        print("No reply-chain messages found in any conversation.")
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

        title = _conv_title(conv_id, nodes, edges_contains)
        depth_dist = " ".join(f"{d}:{depths.count(d)}" for d in sorted(set(depths)))
        conv_metrics.append((max_depth, conv_id, n_msgs, mean_depth, branch_factor, title, depth_dist))

    conv_metrics.sort(key=lambda x: (-x[0], -x[2]))

    print(f"\nConversations with reply chains: {len(conv_metrics)}")
    print()
    print(f"  {'depth':>6s}  {'msgs':>5s}  {'avg_d':>6s}  {'branch':>7s}  {'title':48s}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*6}  {'─'*7}  {'─'*48}")

    for max_depth, conv_id, n_msgs, mean_depth, branch_factor, title, _dist in conv_metrics[:args.top]:
        print(f"  {max_depth:>6d}  {n_msgs:>5d}  {mean_depth:>6.2f}  {branch_factor:>7.3f}  {title:48s}")

    # Depth distribution across all conversations
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
