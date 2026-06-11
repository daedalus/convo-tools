from __future__ import annotations

import math
import pickle
import sqlite3
import threading
from collections import Counter, defaultdict
from typing import TYPE_CHECKING, Any

import networkx as nx

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS node (
    id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    role TEXT DEFAULT '',
    text TEXT DEFAULT '',
    name TEXT DEFAULT '',
    entity_type TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS edge_contains (
    conv_id TEXT NOT NULL,
    msg_id TEXT NOT NULL,
    PRIMARY KEY (conv_id, msg_id)
);

CREATE TABLE IF NOT EXISTS edge_replies_to (
    parent_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    PRIMARY KEY (parent_id, child_id)
);

CREATE TABLE IF NOT EXISTS edge_mentions (
    msg_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    PRIMARY KEY (msg_id, entity_id)
);

CREATE TABLE IF NOT EXISTS edge_cooc (
    entity_a_int INTEGER NOT NULL,
    entity_b_int INTEGER NOT NULL,
    weight INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (entity_a_int, entity_b_int)
);

CREATE TABLE IF NOT EXISTS edge_keyword (
    msg_id TEXT NOT NULL,
    keyword_id TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (msg_id, keyword_id)
);

CREATE TABLE IF NOT EXISTS processed_message (
    id TEXT PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS entity_int (
    int_id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_id TEXT NOT NULL UNIQUE
);

CREATE INDEX IF NOT EXISTS idx_node_label ON node(label);
CREATE INDEX IF NOT EXISTS idx_node_name ON node(name);
CREATE INDEX IF NOT EXISTS idx_node_entity_type ON node(entity_type);
CREATE INDEX IF NOT EXISTS idx_edge_contains_conv ON edge_contains(conv_id);
CREATE INDEX IF NOT EXISTS idx_edge_contains_msg ON edge_contains(msg_id);
CREATE INDEX IF NOT EXISTS idx_edge_mentions_entity ON edge_mentions(entity_id);
CREATE INDEX IF NOT EXISTS idx_edge_mentions_msg ON edge_mentions(msg_id);
CREATE INDEX IF NOT EXISTS idx_edge_keyword_kw ON edge_keyword(keyword_id);
CREATE INDEX IF NOT EXISTS idx_edge_keyword_msg ON edge_keyword(msg_id);
CREATE INDEX IF NOT EXISTS idx_edge_replies_parent ON edge_replies_to(parent_id);
CREATE INDEX IF NOT EXISTS idx_edge_replies_child ON edge_replies_to(child_id);

PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;
PRAGMA cache_size=-65536;
"""


class GraphDB:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._local = threading.local()
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.path, isolation_level=None)
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-65536")
        return self._local.conn

    def _init_db(self) -> None:
        conn = self._conn()
        conn.executescript(SCHEMA_SQL)
        conn.commit()

    def close(self) -> None:
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    # ── Node operations ────────────────────────────────────────────────

    def upsert_node(self, node_id: str, **attrs: Any) -> None:
        conn = self._conn()
        existing = conn.execute(
            "SELECT label, role, text, name, entity_type FROM node WHERE id = ?",
            (node_id,),
        ).fetchone()
        if existing:
            merged = {
                "label": attrs.get("label", existing["label"]),
                "role": attrs.get("role", existing["role"] or ""),
                "text": attrs.get("text", existing["text"] or ""),
                "name": attrs.get("name", existing["name"] or ""),
                "entity_type": attrs.get("entity_type", existing["entity_type"] or ""),
            }
            conn.execute(
                """UPDATE node SET label=?, role=?, text=?, name=?, entity_type=?
                   WHERE id=?""",
                (merged["label"], merged["role"], merged["text"],
                 merged["name"], merged["entity_type"], node_id),
            )
        else:
            conn.execute(
                """INSERT INTO node (id, label, role, text, name, entity_type)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    node_id,
                    attrs.get("label", ""),
                    attrs.get("role", ""),
                    attrs.get("text", ""),
                    attrs.get("name", ""),
                    attrs.get("entity_type", ""),
                ),
            )

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT id, label, role, text, name, entity_type FROM node WHERE id = ?",
            (node_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def node_exists(self, node_id: str) -> bool:
        row = self._conn().execute(
            "SELECT 1 FROM node WHERE id = ?", (node_id,)
        ).fetchone()
        return row is not None

    def search_nodes(
        self,
        label: str = "",
        name_substr: str = "",
        entity_type: str = "",
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if label:
            clauses.append("label = ?")
            params.append(label)
        if name_substr:
            clauses.append("name LIKE ?")
            params.append(f"%{name_substr}%")
        if entity_type:
            clauses.append("entity_type = ?")
            params.append(entity_type.upper())
        where = " AND ".join(clauses) if clauses else "1=1"
        rows = self._conn().execute(
            f"SELECT id, label, role, text, name, entity_type FROM node WHERE {where} LIMIT ?",
            (*params, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_nodes_by_label(self, label: str) -> list[dict[str, Any]]:
        rows = self._conn().execute(
            "SELECT id, label, role, text, name, entity_type FROM node WHERE label = ?",
            (label,),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_nodes_by_label(self) -> dict[str, int]:
        rows = self._conn().execute(
            "SELECT label, COUNT(*) as cnt FROM node GROUP BY label"
        ).fetchall()
        return {r["label"]: r["cnt"] for r in rows}

    def delete_all_nodes(self) -> None:
        self._conn().execute("DELETE FROM node")

    # ── Edge operations ────────────────────────────────────────────────

    def add_edge_contains(self, conv_id: str, msg_id: str) -> None:
        self._conn().execute(
            "INSERT OR IGNORE INTO edge_contains (conv_id, msg_id) VALUES (?, ?)",
            (conv_id, msg_id),
        )

    def add_edge_replies_to(self, parent_id: str, child_id: str) -> None:
        self._conn().execute(
            "INSERT OR IGNORE INTO edge_replies_to (parent_id, child_id) VALUES (?, ?)",
            (parent_id, child_id),
        )

    def add_edge_mentions(self, msg_id: str, entity_id: str) -> None:
        self._conn().execute(
            "INSERT OR IGNORE INTO edge_mentions (msg_id, entity_id) VALUES (?, ?)",
            (msg_id, entity_id),
        )

    def _ensure_entity_int(self, entity_id: str) -> int:
        conn = self._conn()
        conn.execute(
            "INSERT OR IGNORE INTO entity_int (entity_id) VALUES (?)",
            (entity_id,),
        )
        return conn.execute(
            "SELECT int_id FROM entity_int WHERE entity_id = ?",
            (entity_id,),
        ).fetchone()["int_id"]

    def add_edge_cooc(self, entity_a: str, entity_b: str) -> None:
        a_int = self._ensure_entity_int(entity_a)
        b_int = self._ensure_entity_int(entity_b)
        if a_int == b_int:
            return
        a_int, b_int = (a_int, b_int) if a_int <= b_int else (b_int, a_int)
        self._conn().execute(
            "INSERT INTO edge_cooc (entity_a_int, entity_b_int) VALUES (?, ?) "
            "ON CONFLICT(entity_a_int, entity_b_int) DO UPDATE SET weight = weight + 1",
            (a_int, b_int),
        )

    def add_edge_keyword(self, msg_id: str, keyword_id: str, weight: float) -> None:
        self._conn().execute(
            "INSERT OR REPLACE INTO edge_keyword (msg_id, keyword_id, weight) VALUES (?, ?, ?)",
            (msg_id, keyword_id, weight),
        )

    def get_edges_contains(self) -> list[tuple[str, str]]:
        rows = self._conn().execute(
            "SELECT conv_id, msg_id FROM edge_contains"
        ).fetchall()
        return [(r["conv_id"], r["msg_id"]) for r in rows]

    def get_edges_replies_to(self) -> list[tuple[str, str]]:
        rows = self._conn().execute(
            "SELECT parent_id, child_id FROM edge_replies_to"
        ).fetchall()
        return [(r["parent_id"], r["child_id"]) for r in rows]

    def get_edges_mentions(self) -> list[tuple[str, str]]:
        rows = self._conn().execute(
            "SELECT msg_id, entity_id FROM edge_mentions"
        ).fetchall()
        return [(r["msg_id"], r["entity_id"]) for r in rows]

    def get_edges_cooc(self) -> list[tuple[str, str, int]]:
        rows = self._conn().execute(
            "SELECT a.entity_id AS entity_a, b.entity_id AS entity_b, weight "
            "FROM edge_cooc "
            "JOIN entity_int a ON edge_cooc.entity_a_int = a.int_id "
            "JOIN entity_int b ON edge_cooc.entity_b_int = b.int_id"
        ).fetchall()
        return [(r["entity_a"], r["entity_b"], r["weight"]) for r in rows]

    def get_edges_keywords(self) -> list[tuple[str, str, float]]:
        rows = self._conn().execute(
            "SELECT msg_id, keyword_id, weight FROM edge_keyword"
        ).fetchall()
        return [(r["msg_id"], r["keyword_id"], r["weight"]) for r in rows]

    def count_edges(self) -> dict[str, int]:
        return {
            "CONTAINS": self._conn().execute(
                "SELECT COUNT(*) as c FROM edge_contains"
            ).fetchone()["c"],
            "REPLIES_TO": self._conn().execute(
                "SELECT COUNT(*) as c FROM edge_replies_to"
            ).fetchone()["c"],
            "MENTIONS": self._conn().execute(
                "SELECT COUNT(*) as c FROM edge_mentions"
            ).fetchone()["c"],
            "CO_OCCURS_WITH": self._conn().execute(
                "SELECT COUNT(*) as c FROM edge_cooc"
            ).fetchone()["c"],
            "HAS_KEYWORD": self._conn().execute(
                "SELECT COUNT(*) as c FROM edge_keyword"
            ).fetchone()["c"],
        }

    def delete_all_edges(self) -> None:
        for table in ("edge_contains", "edge_replies_to", "edge_mentions",
                      "edge_cooc", "edge_keyword"):
            self._conn().execute(f"DELETE FROM {table}")
        self._conn().execute("DELETE FROM entity_int")

    def clear_all(self) -> None:
        self.delete_all_nodes()
        self.delete_all_edges()

    # ── Processed message tracking (resume-safe incremental build) ────

    def mark_messages_processed(self, msg_ids: set[str]) -> None:
        conn = self._conn()
        conn.execute("BEGIN TRANSACTION")
        try:
            for mid in msg_ids:
                conn.execute(
                    "INSERT OR IGNORE INTO processed_message (id) VALUES (?)",
                    (mid,),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def get_processed_message_ids(self) -> set[str]:
        rows = self._conn().execute("SELECT id FROM processed_message").fetchall()
        return {r["id"] for r in rows}

    def get_unprocessed_message_ids(self) -> set[str]:
        rows = self._conn().execute(
            "SELECT id FROM node WHERE label = 'Message' AND id NOT IN (SELECT id FROM processed_message)"
        ).fetchall()
        return {r["id"] for r in rows}

    # ── Batch operations ───────────────────────────────────────────────

    def add_nodes_batch(self, nodes: dict[str, dict[str, Any]]) -> None:
        conn = self._conn()
        conn.execute("BEGIN TRANSACTION")
        try:
            for nid, attrs in nodes.items():
                self.upsert_node(nid, **attrs)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def add_contains_batch(self, edges: set[tuple[str, str]]) -> None:
        conn = self._conn()
        conn.execute("BEGIN TRANSACTION")
        try:
            for conv_id, msg_id in edges:
                self.add_edge_contains(conv_id, msg_id)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def add_replies_to_batch(self, edges: set[tuple[str, str]]) -> None:
        conn = self._conn()
        conn.execute("BEGIN TRANSACTION")
        try:
            for p, c in edges:
                self.add_edge_replies_to(p, c)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def add_mentions_batch(self, edges: set[tuple[str, str]]) -> None:
        conn = self._conn()
        conn.execute("BEGIN TRANSACTION")
        try:
            for msg_id, eid in edges:
                self.add_edge_mentions(msg_id, eid)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def add_cooc_batch(self, edges: set[tuple[str, str]]) -> None:
        conn = self._conn()
        conn.execute("BEGIN TRANSACTION")
        try:
            for a, b in edges:
                self.add_edge_cooc(a, b)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def add_keywords_batch(
        self, edges: list[tuple[str, str, float]]
    ) -> None:
        conn = self._conn()
        conn.execute("BEGIN TRANSACTION")
        try:
            for msg_id, kid, w in edges:
                self.add_edge_keyword(msg_id, kid, w)
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # ── Incremental batch (build_graph style) ──────────────────────────

    def add_graph_batch(self, graph_data: dict[str, Any]) -> None:
        self.add_nodes_batch(graph_data.get("nodes", {}))
        self.add_contains_batch(graph_data.get("edges_contains", set()))
        self.add_replies_to_batch(graph_data.get("edges_replies_to", set()))
        self.add_mentions_batch(graph_data.get("edges_mentions", set()))
        self.add_cooc_batch(graph_data.get("edges_cooc", set()))
        self.add_keywords_batch(graph_data.get("edges_keywords", []))

    # ── High-level query methods ───────────────────────────────────────

    def get_entity_mention_counts(self) -> Counter[str]:
        rows = self._conn().execute(
            "SELECT entity_id, COUNT(*) as cnt FROM edge_mentions GROUP BY entity_id"
        ).fetchall()
        return Counter({r["entity_id"]: r["cnt"] for r in rows})

    def get_keyword_stats(self) -> dict[str, dict[str, Any]]:
        rows = self._conn().execute(
            """SELECT keyword_id, COUNT(*) as cnt, SUM(weight) as total_weight
               FROM edge_keyword GROUP BY keyword_id"""
        ).fetchall()
        return {
            r["keyword_id"]: {"count": r["cnt"], "total_score": r["total_weight"] or 0.0}
            for r in rows
        }

    def get_msg_to_conv_map(self) -> dict[str, str]:
        rows = self._conn().execute(
            "SELECT msg_id, conv_id FROM edge_contains"
        ).fetchall()
        return {r["msg_id"]: r["conv_id"] for r in rows}

    def get_conv_msgs_map(self) -> dict[str, list[str]]:
        rows = self._conn().execute(
            "SELECT conv_id, msg_id FROM edge_contains ORDER BY msg_id"
        ).fetchall()
        result: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            result[r["conv_id"]].append(r["msg_id"])
        return dict(result)

    def get_entity_messages(self) -> dict[str, set[str]]:
        rows = self._conn().execute(
            "SELECT entity_id, msg_id FROM edge_mentions"
        ).fetchall()
        result: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            result[r["entity_id"]].add(r["msg_id"])
        return dict(result)

    def get_message_keywords(self) -> dict[str, set[str]]:
        rows = self._conn().execute(
            "SELECT msg_id, keyword_id FROM edge_keyword"
        ).fetchall()
        result: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            result[r["msg_id"]].add(r["keyword_id"])
        return dict(result)

    def get_message_entities(self) -> dict[str, set[str]]:
        rows = self._conn().execute(
            "SELECT msg_id, entity_id FROM edge_mentions"
        ).fetchall()
        result: dict[str, set[str]] = defaultdict(set)
        for r in rows:
            result[r["msg_id"]].add(r["entity_id"])
        return dict(result)

    def get_entity_id_set(self) -> set[str]:
        rows = self._conn().execute(
            "SELECT id FROM node WHERE label = 'Entity'"
        ).fetchall()
        return {r["id"] for r in rows}

    def get_message_id_set(self) -> set[str]:
        rows = self._conn().execute(
            "SELECT id FROM node WHERE label = 'Message'"
        ).fetchall()
        return {r["id"] for r in rows}

    # ── Entity co-occurrence graph (NetworkX) ──────────────────────────

    def build_entity_cooc_graph(self) -> nx.Graph:
        g = nx.Graph()
        entity_ids = self.get_entity_id_set()
        g.add_nodes_from(entity_ids)
        rows = self._conn().execute(
            "SELECT a.entity_id AS entity_a, b.entity_id AS entity_b, weight "
            "FROM edge_cooc "
            "JOIN entity_int a ON edge_cooc.entity_a_int = a.int_id "
            "JOIN entity_int b ON edge_cooc.entity_b_int = b.int_id"
        ).fetchall()
        for r in rows:
            if r["entity_a"] in entity_ids and r["entity_b"] in entity_ids:
                g.add_edge(r["entity_a"], r["entity_b"], weight=r["weight"])
        return g

    # ── Reply chain graph (NetworkX) ───────────────────────────────────

    def build_reply_graph(self) -> nx.DiGraph:
        g = nx.DiGraph()
        rows = self._conn().execute(
            "SELECT parent_id, child_id FROM edge_replies_to"
        ).fetchall()
        g.add_edges_from((r["parent_id"], r["child_id"]) for r in rows)
        msg_ids = self.get_message_id_set()
        g.add_nodes_from(msg_ids)
        return g

    # ── Stats ──────────────────────────────────────────────────────────

    def graph_stats(self) -> dict[str, Any]:
        node_counts = self.count_nodes_by_label()
        edge_counts = self.count_edges()
        entity_type_rows = self._conn().execute(
            "SELECT entity_type, COUNT(*) as cnt FROM node WHERE label='Entity' AND entity_type != '' GROUP BY entity_type ORDER BY cnt DESC LIMIT 20"
        ).fetchall()
        role_rows = self._conn().execute(
            "SELECT role, COUNT(*) as cnt FROM node WHERE label='Message' AND role != '' GROUP BY role"
        ).fetchall()
        return {
            "nodes": {
                "total": sum(node_counts.values()),
                "by_label": node_counts,
                "entity_types": {r["entity_type"]: r["cnt"] for r in entity_type_rows},
                "message_roles": {r["role"]: r["cnt"] for r in role_rows},
            },
            "edges": edge_counts,
        }

    # ── Export to pickle (backward compat) ────────────────────────────

    def to_pickle(self) -> dict[str, Any]:
        nodes: dict[str, dict[str, Any]] = {}
        for row in self._conn().execute("SELECT * FROM node").fetchall():
            attrs = {k: row[k] for k in row.keys() if k != "id"}
            attrs = {k: v for k, v in attrs.items() if v}
            attrs.setdefault("label", "")
            nodes[row["id"]] = attrs
        return {
            "nodes": nodes,
            "edges_contains": set(self.get_edges_contains()),
            "edges_replies_to": set(self.get_edges_replies_to()),
            "edges_mentions": set(self.get_edges_mentions()),
            "edges_cooc": set(self.get_edges_cooc()),
            "edges_keywords": self.get_edges_keywords(),
        }

    # ── Bulk import from pickle (migration helper) ─────────────────────

    @classmethod
    def from_pickle(
        cls, pickle_path: str | Path, db_path: str | Path
    ) -> GraphDB:
        with open(pickle_path, "rb") as f:
            data = pickle.load(f)
        db = cls(db_path)
        db.add_graph_batch(data)
        db._conn().commit()
        print(f"Imported {len(data.get('nodes', {}))} nodes into {db_path}")
        return db
