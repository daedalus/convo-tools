from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

import networkx as nx

from convo_tools._graph_db import GraphDB


def _entity_name(entity_id: str, db: GraphDB) -> str:
    node = db.get_node(entity_id)
    if node and node.get("name"):
        return node["name"]
    return entity_id.split("::", 2)[-1]


def _entity_type(entity_id: str, db: GraphDB) -> str:
    node = db.get_node(entity_id)
    if node and node.get("entity_type"):
        return node["entity_type"]
    if "::" in entity_id:
        return entity_id.split("::")[1]
    return ""


def run_topics(db_path: str | Path, args: argparse.Namespace) -> None:
    db = GraphDB(db_path)

    g = db.build_entity_cooc_graph()

    n = g.number_of_nodes()
    m = g.number_of_edges()
    print(f"Entity co-occurrence graph: {n} nodes, {m} edges")

    if n < 2:
        print("Graph too small for community detection.")
        db.close()
        return

    components = sorted(nx.connected_components(g), key=len, reverse=True)
    large_components = [c for c in components if len(c) >= 3]
    print(f"  Connected components: {len(components)} ({len(components) - len(large_components)} too small for Louvain)")

    if not large_components:
        print("No component large enough for meaningful community detection (need >= 3).")
        db.close()
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

    msg_keywords = db.get_message_keywords()
    entity_msgs = db.get_entity_messages()

    display_idx = 0
    for i, comm in enumerate(communities):
        if len(comm) < min_size:
            continue
        display_idx += 1

        comm_g = g.subgraph(comm)
        deg = dict(comm_g.degree())
        top = sorted(deg.items(), key=lambda x: -x[1])[: args.top]

        type_counts: Counter[str] = Counter()
        for eid in comm:
            type_counts[_entity_type(eid, db)] += 1

        print(f"Cluster {display_idx} ({len(comm)} entities, {comm_g.number_of_edges()} internal edges)")
        type_summary = " | ".join(
            f"{t}: {c}" for t, c in type_counts.most_common(5) if t
        )
        if type_summary:
            print(f"  Types: {type_summary}")

        names = []
        for eid, d in top:
            names.append(f"{_entity_name(eid, db)[:30]} (deg={d})")
        print("  Top entities:")
        for line in names:
            print(f"    {line}")

        cluster_msgs: set[str] = set()
        for eid in comm:
            cluster_msgs |= entity_msgs.get(eid, set())

        kw_counter: Counter[str] = Counter()
        for mid in cluster_msgs:
            for kw_id in msg_keywords.get(mid, set()):
                name = _entity_name(kw_id, db)
                if name:
                    kw_counter[name] += 1

        if kw_counter:
            top_kws = kw_counter.most_common(5)
            kw_line = ", ".join(f"{k} ({c})" for k, c in top_kws)
            print(f"  Top keywords: {kw_line}")

        print()

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
                        _entity_name(eid, db),
                        _entity_type(eid, db),
                        row_idx,
                        deg.get(eid, 0),
                    ])
        print(f"Wrote {args.output}")

    db.close()
