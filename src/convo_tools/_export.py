from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

import networkx as nx

from convo_tools._graph_db import GraphDB


def graph_to_gexf(db: GraphDB, output_path: Path) -> None:
    g = nx.DiGraph()

    for label in ("Message", "Entity", "Keyword", "Conversation"):
        for node in db.get_all_nodes_by_label(label):
            node_id = node["id"]
            attrs = {k: v for k, v in node.items() if k != "id" and v}
            g.add_node(node_id, **attrs)

        if label == "Conversation":
            for node in db.get_all_nodes_by_label(label):
                meta = db.get_conv_meta(node["id"])
                if meta:
                    g.nodes[node["id"]].update(meta)

    for src, dst in db.get_edges_contains():
        g.add_edge(src, dst, type="CONTAINS")

    for src, dst in db.get_edges_replies_to():
        g.add_edge(src, dst, type="REPLIES_TO")

    for src, dst in db.get_edges_mentions():
        g.add_edge(src, dst, type="MENTIONS")

    for src, dst, weight in db.get_edges_cooc():
        g.add_edge(src, dst, type="CO_OCCURS_WITH", weight=weight)
        g.add_edge(dst, src, type="CO_OCCURS_WITH", weight=weight)

    for src, dst, weight in db.get_edges_keywords():
        g.add_edge(src, dst, type="HAS_KEYWORD", weight=weight)

    nx.write_gexf(g, str(output_path))
    print(f"Wrote {output_path}")
    print(f"  {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")


def run_export(db_path: str | Path, args: argparse.Namespace) -> None:
    db = GraphDB(db_path)
    try:
        graph_to_gexf(db, args.output)
    finally:
        db.close()
