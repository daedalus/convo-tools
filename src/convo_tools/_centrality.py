from __future__ import annotations

import csv
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

from convo_tools._graph_db import GraphDB


def _entity_name(entity_id: str, db: GraphDB) -> str:
    node = db.get_node(entity_id)
    if node:
        return node.get("name") or entity_id.split("::", 2)[-1]
    return entity_id.split("::", 2)[-1]


def _entity_type(entity_id: str, db: GraphDB) -> str:
    node = db.get_node(entity_id)
    if node:
        return node.get("entity_type") or (entity_id.split("::")[1] if "::" in entity_id else "")
    return entity_id.split("::")[1] if "::" in entity_id else ""


def run_centrality(db_path: str | Path, args: argparse.Namespace) -> None:
    import igraph as ig

    db = GraphDB(db_path)
    g = db.build_entity_cooc_graph(min_weight=args.min_weight)

    n = g.vcount()
    m = g.ecount()
    print(f"Entity co-occurrence graph: {n} nodes, {m} edges")
    print(f"  Connected components: {len(g.components())}")

    if n < 2:
        print("Graph too small for meaningful centrality.")
        db.close()
        return

    largest = max(g.components(), key=len)
    lg = g.subgraph(largest)
    print(f"  Largest component: {lg.vcount()} nodes, {lg.ecount()} edges")

    samples = args.samples
    if samples <= 0:
        samples = n
    samples = min(samples, n)
    exact = args.exact or samples >= n

    import time as _time

    print(f"\nComputing betweenness centrality{' (exact)' if exact else f' (sampled, k={samples})'}...", end="", flush=True)
    sys.stdout.flush()
    _t0 = _time.monotonic()

    betweenness = lg.betweenness() if exact else lg.betweenness(cutoff=samples)
    print(f" {_time.monotonic() - _t0:.1f}s")

    sorted_cent = sorted(enumerate(betweenness), key=lambda x: -x[1])

    print(f"\nTop {args.top} entities by betweenness centrality:")
    print(f"  {'entity':36s} {'type':12s} {'centrality':>12s} {'degree':>6s}")
    print(f"  {'─'*36} {'─'*12} {'─'*12} {'─'*6}")
    for idx, score in sorted_cent[: args.top]:
        entity_id = lg.vs[idx]["name"]
        name = _entity_name(entity_id, db)[:36]
        etype = _entity_type(entity_id, db)[:12]
        deg = lg.degree(idx)
        print(f"  {name:36s} {etype:12s} {score:>12.5f} {deg:>6d}")

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["entity_id", "name", "entity_type", "betweenness", "degree"])
            for idx, score in sorted_cent:
                entity_id = lg.vs[idx]["name"]
                name = _entity_name(entity_id, db)
                etype = _entity_type(entity_id, db)
                deg = lg.degree(idx)
                w.writerow([entity_id, name, etype, f"{score:.6f}", str(deg)])
        print(f"\nWrote {args.output} ({len(sorted_cent)} entities)")

    db.close()
