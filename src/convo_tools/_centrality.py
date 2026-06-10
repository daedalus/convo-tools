from __future__ import annotations

import csv
import pickle
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from typing import Any

import networkx as nx


def _entity_name(entity_id: str, nodes: dict[str, Any]) -> str:
    return nodes.get(entity_id, {}).get("name") or entity_id.split("::", 2)[-1]


def _entity_type(entity_id: str, nodes: dict[str, Any]) -> str:
    return nodes.get(entity_id, {}).get("entity_type") or entity_id.split("::")[1] if "::" in entity_id else ""


def run_centrality(args: argparse.Namespace) -> None:
    with open(args.pickle_path, "rb") as f:
        graph = pickle.load(f)

    if not isinstance(graph, dict) or "nodes" not in graph:
        print("Error: graph pickle missing 'nodes' key", file=sys.stderr)
        return

    nodes = graph.get("nodes", {})
    edges_cooc = graph.get("edges_cooc", set())

    entity_ids = {
        eid for eid in nodes
        if nodes[eid].get("label") == "Entity"
    }

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
    print(f"  Connected components: {nx.number_connected_components(g)}")

    if n < 2:
        print("Graph too small for meaningful centrality.")
        return

    largest = max(nx.connected_components(g), key=len)
    lg = g.subgraph(largest)
    print(f"  Largest component: {lg.number_of_nodes()} nodes, {lg.number_of_edges()} edges")

    samples = args.samples
    if samples <= 0:
        samples = n
    samples = min(samples, n)
    exact = args.exact or samples >= n

    print(f"\nComputing betweenness centrality{' (exact)' if exact else f' (sampled, k={samples})'}...")
    sys.stdout.flush()

    centrality = nx.betweenness_centrality(
        lg,
        k=None if exact else samples,
        normalized=True,
        seed=42,
        endpoints=False,
    )

    sorted_cent = sorted(centrality.items(), key=lambda x: -x[1])

    print(f"\nTop {args.top} entities by betweenness centrality:")
    print(f"  {'entity':36s} {'type':12s} {'centrality':>12s} {'degree':>6s}")
    print(f"  {'─'*36} {'─'*12} {'─'*12} {'─'*6}")
    for entity_id, score in sorted_cent[: args.top]:
        name = _entity_name(entity_id, nodes)[:36]
        etype = _entity_type(entity_id, nodes)[:12]
        deg = lg.degree(entity_id)
        print(f"  {name:36s} {etype:12s} {score:>12.5f} {deg:>6d}")

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["entity_id", "name", "entity_type", "betweenness", "degree"])
            for entity_id, score in sorted_cent:
                name = _entity_name(entity_id, nodes)
                etype = _entity_type(entity_id, nodes)
                deg = lg.degree(entity_id)
                w.writerow([entity_id, name, etype, f"{score:.6f}", str(deg)])
        print(f"\nWrote {args.output} ({len(sorted_cent)} entities)")
