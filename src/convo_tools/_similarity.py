from __future__ import annotations

import csv
import itertools
import sys
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from pathlib import Path
    from typing import Any

from convo_tools._graph_db import GraphDB
from convo_tools._util import safe_pickle_load


def _conv_title(conv_id: str, msgs_by_conv: dict[str, list[dict[str, Any]]]) -> str:
    msgs = msgs_by_conv.get(conv_id, [])
    for m in msgs:
        t = m.get("text", "")
        if isinstance(t, str) and t.strip():
            return t.strip()[:60]
    return conv_id[:16]


def run_similarity(db_path: str | Path, args: argparse.Namespace) -> None:
    import pickle

    db = GraphDB(db_path)

    edges_contains = db.get_edges_contains()
    edges_mentions = db.get_edges_mentions()
    edges_keywords = db.get_edges_keywords()

    messages: list[dict[str, Any]] = safe_pickle_load(args.messages)

    msg_entities: dict[str, set[str]] = defaultdict(set)
    for msg_id, entity_id in edges_mentions:
        msg_entities[msg_id].add(entity_id)

    msg_keywords: dict[str, set[str]] = defaultdict(set)
    for msg_id, keyword_id, _weight in edges_keywords:
        msg_keywords[msg_id].add(keyword_id)

    conv_msgs: dict[str, list[str]] = defaultdict(list)
    for conv_id, msg_id in edges_contains:
        conv_msgs[conv_id].append(msg_id)

    msgs_by_conv: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for m in messages:
        msgs_by_conv[m["conversation_id"]].append(m)

    conv_entities: dict[str, set[str]] = {}
    conv_keywords: dict[str, set[str]] = {}
    for conv_id, msg_ids in conv_msgs.items():
        ents: set[str] = set()
        kwds: set[str] = set()
        for mid in msg_ids:
            ents |= msg_entities.get(mid, set())
            kwds |= msg_keywords.get(mid, set())
        conv_entities[conv_id] = ents
        conv_keywords[conv_id] = kwds

    conv_ids = sorted(conv_entities.keys())
    full_weights = args.all
    min_threshold = args.threshold

    print(f"Conversations: {len(conv_ids)}")
    print(f"Similarity measure: Jaccard on {'entity+keyword' if full_weights else 'entity'} sets")
    print(f"Threshold: {min_threshold}")
    print()

    scored: list[tuple[float, str, str]] = []
    for a, b in itertools.combinations(conv_ids, 2):
        ea = conv_entities[a]
        eb = conv_entities[b]
        if not ea or not eb:
            continue
        ent_j = len(ea & eb) / len(ea | eb)

        if full_weights:
            ka = conv_keywords[a]
            kb = conv_keywords[b]
            kw_j = len(ka & kb) / len(ka | kb) if ka and kb else 0.0
            score = 0.7 * ent_j + 0.3 * kw_j
        else:
            score = ent_j

        if score >= min_threshold:
            scored.append((score, a, b))

    scored.sort(key=lambda x: -x[0])

    if not scored:
        print("No similar conversation pairs found above threshold.")
        db.close()
        return

    print(f"Found {len(scored)} pairs above {min_threshold}")
    print()
    print(f"  {'score':>7s}  {'conv_a':48s}  {'conv_b':48s}")
    print(f"  {'─'*7}  {'─'*48}  {'─'*48}")

    display_n = min(args.top, len(scored))
    for score, a, b in scored[:display_n]:
        ta = _conv_title(a, msgs_by_conv)
        tb = _conv_title(b, msgs_by_conv)
        print(f"  {score:>7.4f}  {ta:48s}  {tb:48s}")

    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["jaccard", "conv_id_a", "conv_id_b", "title_a", "title_b"])
            for score, a, b in scored:
                ta = _conv_title(a, msgs_by_conv)
                tb = _conv_title(b, msgs_by_conv)
                w.writerow([f"{score:.4f}", a, b, ta, tb])
        print(f"\nWrote {args.output} ({len(scored)} pairs)")

    db.close()
