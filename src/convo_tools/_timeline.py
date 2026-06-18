from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse

from convo_tools._graph_db import GraphDB
from convo_tools._util import safe_pickle_load


def _time_bucket(ts: float | None, freq: str) -> str:
    if ts is None:
        return "unknown"
    dt = datetime.fromtimestamp(ts, tz=UTC)
    if freq == "day":
        return dt.strftime("%Y-%m-%d")
    if freq == "week":
        return dt.strftime("%Y-W%V")
    if freq == "year":
        return dt.strftime("%Y")
    return dt.strftime("%Y-%m")


def run_timeline(db_path: str | Path, args: argparse.Namespace) -> None:
    db = GraphDB(db_path)

    messages: list[dict[str, Any]] = safe_pickle_load(args.messages)

    msg_timestamps: dict[str, float | None] = {}
    for m in messages:
        msg_timestamps[m["id"]] = m.get("create_time")

    has_ts = [v for v in msg_timestamps.values() if v is not None]
    if not has_ts:
        print(
            "Warning: no timestamps found in messages. "
            "Re-extract from raw JSON to get timestamps.",
            file=sys.stderr,
        )

    edges_mentions = db.get_edges_mentions()
    bucket_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for msg_id, entity_id in edges_mentions:
        ts = msg_timestamps.get(msg_id)
        bucket = _time_bucket(ts, args.freq)
        bucket_counts[bucket][entity_id] += 1

    sorted_buckets = sorted(b for b in bucket_counts if b != "unknown")
    unknown_bucket = bucket_counts.get("unknown")

    print(f"\nEntity timeline ({args.freq}ly buckets)")
    print(f"Buckets: {sorted_buckets[0] if sorted_buckets else 'N/A'} — {sorted_buckets[-1] if sorted_buckets else 'N/A'} ({len(sorted_buckets)} total)")
    print(f"Entities seen: {len({e for c in bucket_counts.values() for e in c})}")
    print()

    for bucket in sorted_buckets:
        top = bucket_counts[bucket].most_common(args.top)
        parts = []
        for entity_id, count in top:
            node = db.get_node(entity_id)
            name = (node or {}).get("name") or entity_id.split("::", 2)[-1]
            parts.append(f"{name[:28]:28s} {count:4d}")
        print(f"  {bucket}  |  {'  |  '.join(parts)}")

    if unknown_bucket:
        print(f"\n  (unknown timestamp: {sum(unknown_bucket.values())} entity mentions)")

    if args.output:
        overall: Counter[str] = Counter()
        for c in bucket_counts.values():
            overall += c
        top_overall = [eid for eid, _ in overall.most_common(args.top)]

        header_names = []
        for eid in top_overall:
            node = db.get_node(eid)
            name = (node or {}).get("name") or eid.split("::", 2)[-1]
            header_names.append(name)

        with open(args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["bucket"] + header_names)
            for bucket in sorted_buckets:
                row = [bucket]
                for eid in top_overall:
                    row.append(str(bucket_counts[bucket].get(eid, 0)))
                w.writerow(row)

        print(f"\nWrote {args.output} ({len(sorted_buckets)} rows x {len(header_names)} entity columns)")

    db.close()
