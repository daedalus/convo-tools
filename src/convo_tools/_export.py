from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from pathlib import Path
    from typing import Any

import networkx as nx


def graph_to_gexf(graph_data: dict[str, Any], output_path: Path) -> None:
    g = nx.DiGraph()

    for node_id, attrs in graph_data["nodes"].items():
        g.add_node(node_id, **attrs)

    for src, dst in graph_data["edges_contains"]:
        g.add_edge(src, dst, type="CONTAINS")

    for src, dst in graph_data["edges_replies_to"]:
        g.add_edge(src, dst, type="REPLIES_TO")

    for src, dst in graph_data["edges_mentions"]:
        g.add_edge(src, dst, type="MENTIONS")

    for src, dst in graph_data["edges_cooc"]:
        g.add_edge(src, dst, type="CO_OCCURS_WITH")
        g.add_edge(dst, src, type="CO_OCCURS_WITH")

    for src, dst, weight in graph_data["edges_keywords"]:
        g.add_edge(src, dst, type="HAS_KEYWORD", weight=weight)

    nx.write_gexf(g, str(output_path))
    print(f"Wrote {output_path}")
    print(f"  {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")


def run_export(args: argparse.Namespace) -> None:
    pickle_path = args.pickle_path
    output_path = args.output

    with open(pickle_path, "rb") as f:
        graph_data = pickle.load(f)

    if not isinstance(graph_data, dict) or "nodes" not in graph_data:
        print(f"Error: {pickle_path} is not a valid knowledge graph pickle.")
        return

    graph_to_gexf(graph_data, output_path)
