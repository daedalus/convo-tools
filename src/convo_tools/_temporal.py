from __future__ import annotations

import csv
import math
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse
    from typing import Any

from convo_tools._graph_db import GraphDB


def _time_bucket(ts: float | None, window_days: int) -> str:
    if ts is None:
        return "unknown"
    dt = datetime.fromtimestamp(ts, tz=UTC)
    if window_days >= 365:
        return dt.strftime("%Y")
    if window_days >= 28:
        return dt.strftime("%Y-%m")
    if window_days >= 7:
        return dt.strftime("%Y-W%V")
    if window_days >= 1:
        ordinal = dt.toordinal() // window_days
        return f"W{ordinal}"
    return dt.strftime("%Y-%m-%d")


def _entity_name(db: GraphDB, eid: str) -> str:
    n = db.get_node(eid)
    if n is not None:
        name = n.get("name")
        return str(name)[:40] if name else eid.split("::", 2)[-1][:40]
    return eid[:40]


def run_temporal(db_path: str | Path, args: argparse.Namespace) -> None:
    db = GraphDB(db_path)

    with open(args.messages, "rb") as f:
        import pickle
        messages: list[dict[str, Any]] = pickle.load(f)

    msg_timestamps: dict[str, float | None] = {}
    for m in messages:
        msg_timestamps[m["id"]] = m.get("create_time")

    has_ts = sum(1 for v in msg_timestamps.values() if v is not None)
    if has_ts == 0:
        print("No real timestamps found in messages pickle (all create_time are None).")
        print("Using message index as pseudo-time (relative ordering only).")
        for i, m in enumerate(messages):
            msg_timestamps[m["id"]] = float(i * 86400)
        has_ts = len(messages)

    edges_mentions = db.get_edges_mentions()

    #── Bucket entity mentions ──
    window = args.window
    bucket_ents: dict[str, Counter[str]] = defaultdict(Counter)

    for msg_id, ent_id in edges_mentions:
        ts = msg_timestamps.get(msg_id)
        bucket = _time_bucket(ts, window)
        bucket_ents[bucket][ent_id] += 1

    sorted_buckets = sorted(b for b in bucket_ents if b != "unknown")
    if not sorted_buckets:
        print("No timestamped entity mentions found.")
        db.close()
        return

    print(f"  Timespan: {sorted_buckets[0]} — {sorted_buckets[-1]} ({len(sorted_buckets)} buckets, {window}-day windows)")
    n_entities = len({e for c in bucket_ents.values() for e in c})
    n_mentions = sum(sum(c.values()) for c in bucket_ents.values())
    print(f"  Entities: {n_entities}, entity-mentions: {n_mentions}")
    print()

    #── Per-entity metrics ──
    all_entity_ids = {e for c in bucket_ents.values() for e in c}
    entity_metrics: dict[str, dict[str, Any]] = {}

    for eid in all_entity_ids:
        bucket_cts: list[tuple[str, int]] = []
        for b in sorted_buckets:
            count = bucket_ents[b].get(eid, 0)
            if count:
                bucket_cts.append((b, count))

        first_bucket = bucket_cts[0][0] if bucket_cts else None
        last_bucket = bucket_cts[-1][0] if bucket_cts else None
        total = sum(c for _, c in bucket_cts)
        n_active = len(bucket_cts)
        mention_days = [c for _, c in bucket_cts]

        mean_ = total / max(len(sorted_buckets), 1)
        if len(sorted_buckets) > 1 and mean_ > 0:
            variance = sum((c - mean_) ** 2 for c in mention_days) / len(sorted_buckets)
            bursts = sum(1 for c in mention_days if mean_ > 0 and c > mean_ + 2.0 * math.sqrt(variance))
            cv = math.sqrt(variance) / mean_
        else:
            bursts = 0
            cv = 0.0

        entity_metrics[eid] = {
            "first": first_bucket,
            "last": last_bucket,
            "total": total,
            "active_buckets": n_active,
            "bursts": bursts,
            "cv": cv,
        }

    #── Active entity count per bucket ──
    bucket_active_counts: list[tuple[str, int, int]] = []
    for b in sorted_buckets:
        ment = sum(bucket_ents[b].values())
        n_act = len(bucket_ents[b])
        bucket_active_counts.append((b, n_act, ment))

    print("═══ Bucket activity overview ═══")
    print(f"  {'bucket':>12s}  {'entities':>9s}  {'mentions':>9s}")
    print(f"  {'─'*12}  {'─'*9}  {'─'*9}")
    for b, ac, mc in bucket_active_counts:
        print(f"  {b:>12s}  {ac:>9d}  {mc:>9d}")

    #── Top entities by each metric ──
    top_n = args.top

    print()
    print(f"═══ Entity temporal metrics (top {top_n}) ═══")
    print(f"  {'name':30s}  {'total':>6s}  {'first':>12s}  {'last':>12s}  {'active':>7s}  {'bursts':>7s}  {'cv':>6s}")
    print(f"  {'─'*30}  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*7}  {'─'*7}  {'─'*6}")

    sorted_by_total = sorted(entity_metrics.items(), key=lambda x: -x[1]["total"])
    for eid, m in sorted_by_total[:top_n]:
        name = _entity_name(db, eid)
        print(f"  {name:30s}  {m['total']:>6d}  {str(m['first'] or '?'):>12s}  {str(m['last'] or '?'):>12s}  {m['active_buckets']:>7d}  {m['bursts']:>7d}  {m['cv']:>6.3f}")

    #── Most bursty (highest CV, excluding low-total) ──
    bursty = [(eid, m) for eid, m in entity_metrics.items() if m["total"] >= top_n and m["cv"] > 0]
    bursty.sort(key=lambda x: -x[1]["cv"])

    if bursty:
        print()
        print(f"═══ Most bursty entities (CV rank, min {top_n} mentions) ═══")
        print(f"  {'name':30s}  {'cv':>6s}  {'total':>6s}  {'bursts':>7s}  {'active':>7s}")
        print(f"  {'─'*30}  {'─'*6}  {'─'*6}  {'─'*7}  {'─'*7}")
        for eid, m in bursty[:top_n]:
            name = _entity_name(db, eid)
            print(f"  {name:30s}  {m['cv']:>6.3f}  {m['total']:>6d}  {m['bursts']:>7d}  {m['active_buckets']:>7d}")

    #── Entity activity timeline for top entities ──
    print()
    print("═══ Entity activity heatmap (top by total, across all buckets) ═══")
    top_entities = [eid for eid, _ in sorted_by_total[:top_n]]

    name_col_width = 30
    bucket_width = max(5, (120 - name_col_width - 2) // len(sorted_buckets))
    bucket_label = max(5, bucket_width)

    header = " " * (name_col_width + 2)
    for b in sorted_buckets:
        short = b[-bucket_label:] if len(b) > bucket_label else b
        header += f"{short:>{bucket_width}s}"
    print(header)

    for eid in top_entities:
        name = _entity_name(db, eid)
        line = f"{name:>{name_col_width}s}  "
        max_c = max(bucket_ents[b].get(eid, 0) for b in sorted_buckets)
        for b in sorted_buckets:
            c = bucket_ents[b].get(eid, 0)
            if max_c > 0:
                bar = "█" * int(bucket_width * c / max_c)
            else:
                bar = ""
            line += f"{bar:>{bucket_width}s}"
        print(line)

    #── Co-activation: entity pairs co-appearing in same bucket ──
    if args.co_activation:
        print()
        print("═══ Top co-activating entity pairs ═══")

        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        for b in sorted_buckets:
            ents = sorted(bucket_ents[b].keys())
            for i in range(len(ents)):
                e1 = ents[i]
                for j in range(i + 1, len(ents)):
                    e2 = ents[j]
                    pair_counts[(e1, e2)] += 1

        if pair_counts:
            ranked = sorted(pair_counts.items(), key=lambda x: -x[1])

            print(f"  {'buckets':>7s}  {'entity1':30s}  {'entity2':30s}")
            print(f"  {'─'*7}  {'─'*30}  {'─'*30}")
            for (e1, e2), count in ranked[:args.top]:
                n1 = _entity_name(db, e1)
                n2 = _entity_name(db, e2)
                pct = 100.0 * count / len(sorted_buckets)
                print(f"  {count:>7d}  {n1:30s}  {n2:30s}  ({pct:.0f}% of buckets)")
        else:
            print("  No co-activated pairs found.")

    #── CSV output ──
    if args.output:
        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["entity_id", "name", "total_mentions", "first_bucket", "last_bucket",
                       "active_buckets", "burst_count", "cv"])
            for eid, m in sorted_by_total:
                w.writerow([eid, _entity_name(db, eid), m["total"],
                          m["first"] or "", m["last"] or "", m["active_buckets"],
                          m["bursts"], f"{m['cv']:.4f}"])
        print(f"\nWrote {args.output} ({len(sorted_by_total)} entities)")

    db.close()
