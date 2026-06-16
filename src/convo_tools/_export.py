from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

from convo_tools._graph_db import GraphDB


def graph_to_gexf(db: GraphDB, output_path: Path) -> None:
    gexf = ET.Element("gexf", xmlns="http://www.gexf.net/1.3draft", version="1.3")
    graph_el = ET.SubElement(gexf, "graph", defaultedgetype="directed", mode="static")

    nodes_el = ET.SubElement(graph_el, "nodes")
    edges_el = ET.SubElement(graph_el, "edges")

    node_idx: dict[str, int] = {}
    edge_counter = 0

    for label in ("Message", "Entity", "Keyword", "Conversation"):
        for node in db.get_all_nodes_by_label(label):
            node_id = node["id"]
            idx = len(node_idx)
            node_idx[node_id] = idx
            attrs = {k: v for k, v in node.items() if k != "id" and v}
            node_el = ET.SubElement(nodes_el, "node", id=str(idx), label=node_id)
            if attrs:
                attvalues = ET.SubElement(node_el, "attvalues")
                for k, v in attrs.items():
                    ET.SubElement(attvalues, "attvalue", for_=k, value=str(v))

        if label == "Conversation":
            for node in db.get_all_nodes_by_label("Conversation"):
                meta = db.get_conv_meta(node["id"])
                if meta and node["id"] in node_idx:
                    idx = node_idx[node["id"]]
                    node_el = nodes_el.find(f"node[@id='{idx}']")
                    if node_el is not None:
                        attvalues = node_el.find("attvalues")
                        if attvalues is None:
                            attvalues = ET.SubElement(node_el, "attvalues")
                        for k, v in meta.items():
                            if v:
                                ET.SubElement(attvalues, "attvalue", for_=k, value=str(v))

    def _add_edge(src: str, dst: str, **attrs: str) -> None:
        if src in node_idx and dst in node_idx:
            edge_el = ET.SubElement(edges_el, "edge",
                id=str(edge_counter),
                source=str(node_idx[src]),
                target=str(node_idx[dst]))
            edge_counter += 1
            if attrs:
                attvalues = ET.SubElement(edge_el, "attvalues")
                for k, v in attrs.items():
                    ET.SubElement(attvalues, "attvalue", for_=k, value=str(v))

    for src, dst in db.get_edges_contains():
        _add_edge(src, dst, type="CONTAINS")

    for src, dst in db.get_edges_replies_to():
        _add_edge(src, dst, type="REPLIES_TO")

    for r in db.get_edges_mentions():
        _add_edge(r["msg_id"], r["entity_id"], type="MENTIONS")

    for r in db.get_edges_cooc():
        _add_edge(r["entity_a"], r["entity_b"], type="CO_OCCURS_WITH", weight=str(r["weight"]))
        _add_edge(r["entity_b"], r["entity_a"], type="CO_OCCURS_WITH", weight=str(r["weight"]))

    for r in db.get_edges_keywords():
        _add_edge(r["msg_id"], r["keyword_id"], type="HAS_KEYWORD", weight=str(r["weight"]))

    tree = ET.ElementTree(gexf)
    ET.indent(tree, space="  ")
    tree.write(str(output_path), xml_declaration=True, encoding="UTF-8")
    print(f"Wrote {output_path}")
    print(f"  {len(node_idx)} nodes, {edge_counter} edges")


def run_export(db_path: str | Path, args: argparse.Namespace) -> None:
    db = GraphDB(db_path)
    try:
        graph_to_gexf(db, args.output)
    finally:
        db.close()
