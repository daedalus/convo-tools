from __future__ import annotations

import csv
import pickle
import sys
from collections import Counter
from typing import TYPE_CHECKING

from convo_tools._util import safe_pickle_load

if TYPE_CHECKING:
    import argparse
    from typing import Any


def _label_of(nodes: dict[str, Any], nid: str) -> str:
    n = nodes.get(nid)
    if isinstance(n, dict):
        label = n.get("label")
        return str(label) if label is not None else ""
    return ""


def _name_of(nodes: dict[str, Any], nid: str) -> str:
    n = nodes.get(nid)
    if isinstance(n, dict):
        name = n.get("name")
        return str(name) if name is not None else nid[:40]
    return nid[:40]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


def _pct(a: float, b: float) -> str:
    d = b or a
    return f"{100.0 * a / d:.1f}%" if d else "-"


def _diff_named_sets(
    left_set: set[str],
    right_set: set[str],
    label: str,
) -> None:
    only_left = left_set - right_set
    only_right = right_set - left_set
    both = left_set & right_set

    print()
    print(f"  {label}")
    print(f"    shared: {_plural(len(both), 'item')}  ({_pct(len(both), len(left_set | right_set))} of total)")
    print(f"    left-only:  {_plural(len(only_left), 'item')}  ({_pct(len(only_left), len(left_set))} of left)")
    print(f"    right-only: {_plural(len(only_right), 'item')}  ({_pct(len(only_right), len(right_set))} of right)")


def run_diff(args: argparse.Namespace) -> None:
    g_left = _load(str(args.left), "left")
    if g_left is None:
        return
    g_right = _load(str(args.right), "right")
    if g_right is None:
        return

    left_label = args.left_label or "left"
    right_label = args.right_label or "right"

    left_nodes = g_left.get("nodes", {})
    right_nodes = g_right.get("nodes", {})

    #── Node labels ──
    def _node_ids_by_label(nodes: dict[str, Any], label: str) -> set[str]:
        return {nid for nid, attrs in nodes.items() if _label_of(nodes, nid) == label}

    left_conv = _node_ids_by_label(left_nodes, "Conversation")
    right_conv = _node_ids_by_label(right_nodes, "Conversation")
    left_msg = _node_ids_by_label(left_nodes, "Message")
    right_msg = _node_ids_by_label(right_nodes, "Message")
    left_entities = {nid for nid, attrs in left_nodes.items() if attrs.get("label") not in ("Conversation", "Message", "Keyword")}
    right_entities = {nid for nid, attrs in right_nodes.items() if attrs.get("label") not in ("Conversation", "Message", "Keyword")}
    left_kw = {nid for nid, attrs in left_nodes.items() if attrs.get("label") == "Keyword"}
    right_kw = {nid for nid, attrs in right_nodes.items() if attrs.get("label") == "Keyword"}

    #── Edge sets ──
    left_ec = g_left.get("edges_contains", set())
    right_ec = g_right.get("edges_contains", set())
    left_er = g_left.get("edges_replies_to", set())
    right_er = g_right.get("edges_replies_to", set())
    left_em = g_left.get("edges_mentions", set())
    right_em = g_right.get("edges_mentions", set())
    left_ecc = g_left.get("edges_cooc", set())
    right_ecc = g_right.get("edges_cooc", set())
    left_ek = g_left.get("edges_keywords", [])
    right_ek = g_right.get("edges_keywords", [])

    #── Overall summary ──
    print()
    print(f"  {'':>30s}  {left_label:>20s}  {right_label:>20s}  diff")
    print(f"  {'─'*30}  {'─'*20}  {'─'*20}  {'─'*8}")
    overall = [
        ("Conversations", len(left_conv), len(right_conv)),
        ("Messages", len(left_msg), len(right_msg)),
        ("Entities", len(left_entities), len(right_entities)),
        ("Keywords", len(left_kw), len(right_kw)),
        ("Edges contains", len(left_ec), len(right_ec)),
        ("Edges replies_to", len(left_er), len(right_er)),
        ("Edges mentions", len(left_em), len(right_em)),
        ("Edges cooc", len(left_ecc), len(right_ecc)),
        ("Edges keywords", len(left_ek), len(right_ek)),
    ]
    for name, ln, rn in overall:
        d = ln - rn
        sign = "+" if d > 0 else ""
        print(f"  {name:>30s}:  {ln:>20d}  {rn:>20d}  {sign}{d}")

    #── Set overlaps ──
    print()
    print("═══ Set overlap analysis ═══")

    _diff_named_sets(left_entities, right_entities, "Entities")
    _diff_named_sets(left_kw, right_kw, "Keywords")
    _diff_named_sets(left_conv, right_conv, "Conversations")

    print()
    print("═══ Entity mention frequency comparison ═══")

    left_mention_counts: Counter[str] = Counter()
    for msg_id, ent_id in left_em:
        left_mention_counts[ent_id] += 1
    right_mention_counts: Counter[str] = Counter()
    for msg_id, ent_id in right_em:
        right_mention_counts[ent_id] += 1

    all_entities = left_entities | right_entities
    entity_rows: list[tuple[int, str, str, int, int]] = []
    for eid in all_entities:
        lc = left_mention_counts.get(eid, 0)
        rc = right_mention_counts.get(eid, 0)
        name = _name_of(left_nodes if eid in left_nodes else right_nodes, eid)
        entity_rows.append((abs(lc - rc), eid, name, lc, rc))

    entity_rows.sort(key=lambda x: -x[0])
    top_n = min(args.top, len(entity_rows))

    print(f"\n  Top {top_n} entities with largest mention-count difference:")
    print(f"  {'diff':>6s}  {'name':30s}  {f'{left_label}':>10s}  {f'{right_label}':>10s}  {'lean':>12s}")
    print(f"  {'─'*6}  {'─'*30}  {'─'*10}  {'─'*10}  {'─'*12}")
    for diff, eid, name, lc, rc in entity_rows[:top_n]:
        lean = left_label if lc > rc else right_label if rc > lc else "equal"
        print(f"  {diff:>6d}  {name:30s}  {lc:>10d}  {rc:>10d}  {lean:>12s}")

    #── Entity Jaccard ──
    intersection = left_entities & right_entities
    union = left_entities | right_entities
    jaccard = len(intersection) / len(union) if union else 0.0
    print(f"\n  Entity set Jaccard similarity: {jaccard:.4f}")

    kw_intersection = left_kw & right_kw
    kw_union = left_kw | right_kw
    kw_jaccard = len(kw_intersection) / len(kw_union) if kw_union else 0.0
    print(f"  Keyword set Jaccard similarity: {kw_jaccard:.4f}")

    conv_intersection = left_conv & right_conv
    conv_union = left_conv | right_conv
    conv_jaccard = len(conv_intersection) / len(conv_union) if conv_union else 0.0
    print(f"  Conversation set Jaccard similarity: {conv_jaccard:.4f}")

    #── Per-entity type breakdown ──
    left_entity_types = _entity_type_breakdown(left_nodes, left_entities, left_mention_counts)
    right_entity_types = _entity_type_breakdown(right_nodes, right_entities, right_mention_counts)

    print()
    print("═══ Entity type breakdown ═══")
    print(f"  {'type':>20s}  {f'{left_label}':>10s}  {f'{right_label}':>10s}  mentions(L)  mentions(R)")
    print(f"  {'─'*20}  {'─'*10}  {'─'*10}  {'─'*12}  {'─'*12}")
    all_types = sorted(left_entity_types.keys() | right_entity_types.keys())
    for t in all_types:
        lc = left_entity_types.get(t, 0)
        rc = right_entity_types.get(t, 0)
        lm = sum(v for eid, v in left_mention_counts.items() if _label_of(left_nodes if eid in left_nodes else right_nodes, eid) == t)
        rm = sum(v for eid, v in right_mention_counts.items() if _label_of(right_nodes if eid in right_nodes else left_nodes, eid) == t)
        print(f"  {t:>20s}:  {lc:>10d}  {rc:>10d}  {lm:>12d}  {rm:>12d}")

    #── CSV output ──
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["eid", "name", f"{left_label}_mentions", f"{right_label}_mentions", "difference", "lean"])
            for diff, eid, name, lc, rc in entity_rows:
                lean = left_label if lc > rc else right_label if rc > lc else "" if lc == 0 and rc == 0 else "equal"
                w.writerow([eid, name, lc, rc, diff, lean])
        print(f"\nWrote {args.output} ({len(entity_rows)} entities)")


def _entity_type_breakdown(
    nodes: dict[str, Any],
    entity_ids: set[str],
    mention_counts: Counter[str],
) -> Counter[str]:
    types: Counter[str] = Counter()
    for eid in entity_ids:
        label = _label_of(nodes, eid)
        types[label] += mention_counts.get(eid, 0)
    return types


def _load(path: str, side: str) -> dict[str, Any] | None:
    try:
        data = safe_pickle_load(path)
    except FileNotFoundError:
        print(f"Error: {side} pickle not found: {path}", file=sys.stderr)
        return None
    if not isinstance(data, dict) or "nodes" not in data:
        print(f"Error: {side} pickle is not a valid graph dict (missing 'nodes')", file=sys.stderr)
        return None
    return data
