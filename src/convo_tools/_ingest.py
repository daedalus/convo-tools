from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from typing import Any

from convo_tools._graph_db import GraphDB

KUZU_SCHEMA_SQL = """
CREATE NODE TABLE IF NOT EXISTS Conversation (
    id STRING, label STRING, PRIMARY KEY (id)
);
CREATE NODE TABLE IF NOT EXISTS Message (
    id STRING, label STRING, role STRING, text STRING, PRIMARY KEY (id)
);
CREATE NODE TABLE IF NOT EXISTS Entity (
    id STRING, label STRING, name STRING, entity_type STRING, PRIMARY KEY (id)
);
CREATE NODE TABLE IF NOT EXISTS Keyword (
    id STRING, label STRING, name STRING, PRIMARY KEY (id)
);
CREATE REL TABLE IF NOT EXISTS CONTAINS (FROM Conversation TO Message);
CREATE REL TABLE IF NOT EXISTS REPLIES_TO (FROM Message TO Message);
CREATE REL TABLE IF NOT EXISTS MENTIONS (FROM Message TO Entity);
CREATE REL TABLE IF NOT EXISTS CO_OCCURS_WITH (FROM Entity TO Entity, weight INT32 DEFAULT 1);
CREATE REL TABLE IF NOT EXISTS HAS_KEYWORD (FROM Message TO Keyword, weight DOUBLE);
"""


def _node_rows(nodes: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {
        "Conversation": [],
        "Message": [],
        "Entity": [],
        "Keyword": [],
    }
    for nid, attrs in nodes.items():
        label = attrs.get("label", "")
        if label == "Conversation":
            rows["Conversation"].append({"id": nid, "label": label})
        elif label == "Message":
            rows["Message"].append({
                "id": nid, "label": label, "role": attrs.get("role", ""),
                "text": attrs.get("text", ""),
            })
        elif label == "Entity":
            rows["Entity"].append({
                "id": nid, "label": label, "name": attrs.get("name", ""),
                "entity_type": attrs.get("entity_type", ""),
            })
        elif label == "Keyword":
            rows["Keyword"].append({
                "id": nid, "label": label, "name": attrs.get("name", ""),
            })
    return rows


def _edge_rows(edges: set) -> list[dict[str, str]]:
    rows = []
    for e in edges:
        a, b = e[0], e[1]
        rows.append({"from": a, "to": b})
    return rows


def _kw_edge_rows(edges: set) -> list[dict[str, Any]]:
    rows = []
    for e in edges:
        a, b = e[0], e[1]
        w = e[2] if len(e) > 2 else 1
        rows.append({"from": a, "to": b, "weight": w})
    return rows


def graph_to_kuzu(graph_data: dict[str, Any], db_path: str, overwrite: bool = False) -> None:
    try:
        import kuzu  # noqa: PLC0415, F811
    except ImportError:
        print("Error: kuzu is not installed. Run: pip install kuzu", file=sys.stderr)
        sys.exit(1)

    if overwrite and os.path.exists(db_path):
        os.remove(db_path)

    db = kuzu.Database(db_path)
    conn = kuzu.Connection(db)

    for stmt in KUZU_SCHEMA_SQL.strip().split(";"):
        s = stmt.strip()
        if s:
            conn.execute(s + ";")

    nodes = graph_data.get("nodes", {})
    node_rows = _node_rows(nodes)

    for table, cols in [
        ("Conversation", ["id", "label"]),
        ("Message", ["id", "label", "role", "text"]),
        ("Entity", ["id", "label", "name", "entity_type"]),
        ("Keyword", ["id", "label", "name"]),
    ]:
        rows = node_rows[table]
        if not rows:
            continue
        col_expr = ", ".join(f"{c}: r.{c}" for c in cols)
        conn.execute(
            f"UNWIND $batch AS r CREATE (:{table} {{ {col_expr} }})",
            {"batch": rows},
        )
        print(f"  {table}: {len(rows)} nodes")

    rel_configs: list[tuple[str, str, str, list[dict[str, Any]], bool]] = [
        ("CONTAINS", "Conversation", "Message", _edge_rows(graph_data.get("edges_contains", set())), False),
        ("REPLIES_TO", "Message", "Message", _edge_rows(graph_data.get("edges_replies_to", set())), False),
        ("MENTIONS", "Message", "Entity", _edge_rows(graph_data.get("edges_mentions", set())), False),
        ("CO_OCCURS_WITH", "Entity", "Entity", _kw_edge_rows(graph_data.get("edges_cooc", set())), True),
        ("HAS_KEYWORD", "Message", "Keyword", _kw_edge_rows(graph_data.get("edges_keywords", [])), True),
    ]

    for name, src, dst, rows, has_weight in rel_configs:
        if not rows:
            continue
        if has_weight:
            batch_params = [{"from": r["from"], "to": r["to"]} for r in rows]
            conn.execute(
                f"""
                UNWIND $batch AS r
                MATCH (s:{src}), (t:{dst})
                WHERE s.id = r.from AND t.id = r.to
                CREATE (s)-[:{name}]->(t)
                """,
                {"batch": batch_params},
            )
            for r in rows:
                conn.execute(
                    f"""
                    MATCH (s:{src})-[k:{name}]->(t:{dst})
                    WHERE s.id = $from AND t.id = $to
                    SET k.weight = $weight
                    """,
                    {"from": r["from"], "to": r["to"], "weight": r["weight"]},
                )
        else:
            conn.execute(
                f"""
                UNWIND $batch AS r
                MATCH (s:{src}), (t:{dst})
                WHERE s.id = r.from AND t.id = r.to
                CREATE (s)-[:{name}]->(t)
                """,
                {"batch": rows},
            )
        print(f"  {name}: {len(rows)} edges")

    print(f"\nIngested to {db_path}")


def run_ingest(args: argparse.Namespace) -> None:
    if args.pickle_path:
        import pickle
        with open(args.pickle_path, "rb") as f:
            graph_data = pickle.load(f)
    else:
        db = GraphDB(args.db)
        graph_data = db.to_pickle()
        db.close()

    if not isinstance(graph_data, dict) or "nodes" not in graph_data:
        print("Error: graph data missing 'nodes' key", file=sys.stderr)
        return

    graph_to_kuzu(graph_data, str(args.kuzu_path), overwrite=args.overwrite)
