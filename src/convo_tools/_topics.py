from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse

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
    import igraph as ig

    db = GraphDB(db_path)

    g = db.build_entity_cooc_graph(min_weight=args.min_weight)

    n = g.vcount()
    m = g.ecount()
    print(f"Entity co-occurrence graph: {n} nodes, {m} edges")

    if n < 2:
        print("Graph too small for community detection.")
        db.close()
        return

    components = sorted(g.components(), key=len, reverse=True)
    large_components = [c for c in components if len(c) >= 3]
    print(f"  Connected components: {len(components)} ({len(components) - len(large_components)} too small for Louvain)")

    if not large_components:
        print("No component large enough for meaningful community detection (need >= 3).")
        db.close()
        return

    communities: list[list[int]] = []
    for comp in large_components:
        sg = g.subgraph(comp)
        print(f"\n  Running Louvain on component with {sg.vcount()} nodes...")
        sys.stdout.flush()
        try:
            comms_result = sg.community_multilevel(weights=sg.es["weight"], return_levels=False)
            communities.extend([list(c) for c in comms_result])
        except Exception:
            communities.append(list(comp))

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
        deg = {v["name"]: comm_g.degree(v.index) for v in comm_g.vs}
        top = sorted(deg.items(), key=lambda x: -x[1])[: args.top]

        type_counts: Counter[str] = Counter()
        for v in comm_g.vs:
            type_counts[_entity_type(v["name"], db)] += 1

        print(f"Cluster {display_idx} ({len(comm)} entities, {comm_g.ecount()} internal edges)")
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
        for v in comm_g.vs:
            cluster_msgs |= entity_msgs.get(v["name"], set())

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
                deg = {v["name"]: comm_g.degree(v.index) for v in comm_g.vs}
                for v in comm_g.vs:
                    eid = v["name"]
                    w.writerow([
                        eid,
                        _entity_name(eid, db),
                        _entity_type(eid, db),
                        row_idx,
                        deg.get(eid, 0),
                    ])
        print(f"Wrote {args.output}")

    db.close()
