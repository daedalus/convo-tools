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


def _entity_name(entity_id: str, nodes: dict[str, Any]) -> str:
    return nodes.get(entity_id, {}).get("name") or entity_id.split("::", 2)[-1]


def _entity_type(entity_id: str, nodes: dict[str, Any]) -> str:
    return nodes.get(entity_id, {}).get("entity_type") or (
        entity_id.split("::")[1] if "::" in entity_id else ""
    )


def run_topics(args: argparse.Namespace) -> None:
    with open(args.pickle_path, "rb") as f:
        graph = pickle.load(f)

    if not isinstance(graph, dict) or "nodes" not in graph:
        print("Error: graph pickle missing 'nodes' key", file=sys.stderr)
        return

    nodes = graph.get("nodes", {})
    edges_cooc = graph.get("edges_cooc", set())
    edges_mentions = graph.get("edges_mentions", set())
    edges_keywords = graph.get("edges_keywords", [])

    entity_ids = {nid for nid, attrs in nodes.items() if attrs.get("label") == "Entity"}

    g = nx.Graph()
    g.add_nodes_from(entity_ids)

    edge_count = 0
    for a, b in edges_cooc:
        if a in entity_ids and b in entity_ids:
            g.add_edge(a, b)
            edge_count += 1

    n = g.number_of_nodes()
    m = g.number_of_edges()
    print(f"Entity co-occurrence graph: {n} nodes, {m} edges")

    if n < 2:
        print("Graph too small for community detection.")
        return

    components = sorted(nx.connected_components(g), key=len, reverse=True)
    large_components = [c for c in components if len(c) >= 3]
    print(f"  Connected components: {len(components)} ({len(components) - len(large_components)} too small for Louvain)")

    if not large_components:
        print("No component large enough for meaningful community detection (need >= 3).")
        return

    communities: list[frozenset[str]] = []
    for comp in large_components:
        sg = g.subgraph(comp)
        print(f"\n  Running Louvain on component with {sg.number_of_nodes()} nodes...")
        sys.stdout.flush()
        comms = nx.community.louvain_communities(sg, seed=42)
        communities.extend(comms)

    communities = sorted(communities, key=len, reverse=True)
    print(f"\n  Found {len(communities)} communities across all components")

    print()
    min_size = args.min_size

    # ── Precompute per-message keyword sets for TF-IDF cross-ref ──
    msg_keywords: dict[str, set[str]] = defaultdict(set)
    for msg_id, kw_id, _w in edges_keywords:
        msg_keywords[msg_id].add(kw_id)

    entity_msgs: dict[str, set[str]] = defaultdict(set)
    for msg_id, eid in edges_mentions:
        entity_msgs[eid].add(msg_id)

    display_idx = 0
    for i, comm in enumerate(communities):
        if len(comm) < min_size:
            continue
        display_idx += 1

        # Top entities by degree within cluster
        comm_g = g.subgraph(comm)
        deg = dict(comm_g.degree())
        top = sorted(deg.items(), key=lambda x: -x[1])[: args.top]

        # Entity type distribution
        type_counts: Counter[str] = Counter()
        for eid in comm:
            type_counts[_entity_type(eid, nodes)] += 1

        print(f"Cluster {display_idx} ({len(comm)} entities, {comm_g.number_of_edges()} internal edges)")
        type_summary = " | ".join(
            f"{t}: {c}" for t, c in type_counts.most_common(5) if t
        )
        if type_summary:
            print(f"  Types: {type_summary}")

        # Entity names
        names = []
        for eid, d in top:
            names.append(f"{_entity_name(eid, nodes)[:30]} (deg={d})")
        print("  Top entities:")
        for line in names:
            print(f"    {line}")

        # Cross-ref: characteristic keywords
        cluster_msgs: set[str] = set()
        for eid in comm:
            cluster_msgs |= entity_msgs.get(eid, set())

        kw_counter: Counter[str] = Counter()
        for mid in cluster_msgs:
            for kw_id in msg_keywords.get(mid, set()):
                name = _entity_name(kw_id, nodes)
                if name:
                    kw_counter[name] += 1

        if kw_counter:
            top_kws = kw_counter.most_common(5)
            kw_line = ", ".join(f"{k} ({c})" for k, c in top_kws)
            print(f"  Top keywords: {kw_line}")

        print()

    # ── CSV output ──
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["entity_id", "name", "entity_type", "cluster", "degree_in_cluster"])
            row_idx = 0
            for i, comm in enumerate(communities):
                if len(comm) < min_size:
                    continue
                row_idx += 1
                comm_g = g.subgraph(comm)
                deg = dict(comm_g.degree())
                for eid in sorted(comm):
                    w.writerow([
                        eid,
                        _entity_name(eid, nodes),
                        _entity_type(eid, nodes),
                        row_idx,
                        deg.get(eid, 0),
                    ])
        print(f"Wrote {args.output}")
