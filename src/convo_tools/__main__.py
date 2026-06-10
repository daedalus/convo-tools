from __future__ import annotations

import argparse
import sys
from pathlib import Path

from convo_tools._builder import run_graph
from convo_tools._export import run_export
from convo_tools._extract import run_extract
from convo_tools._join import run_join
from convo_tools._split import run_split
from convo_tools._timeline import run_timeline


def _build_base_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m",
        description="Conversation analysis toolkit.",
    )
    ap.add_argument(
        "-m", "--mode", required=True,
        choices=["extract", "graph", "full", "join", "split", "export", "timeline"],
        help="Operation mode",
    )
    return ap


def _build_extract_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m extract",
        description="Read JSON conversation exports, deduplicate, and save as pickle.",
    )
    ap.add_argument("json_dir", type=Path, help="Directory containing .json files")
    ap.add_argument(
        "pickle_path", nargs="?", type=Path, default=Path("messages.pkl"),
        help="Output pickle path (default: messages.pkl)",
    )
    return ap


def _build_graph_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m graph",
        description="Build a knowledge graph from a pickle of extracted messages.",
    )
    ap.add_argument(
        "pickle_path", nargs="?", type=Path, default=Path("messages.pkl"),
        help="Input pickle path (default: messages.pkl)",
    )
    ap.add_argument(
        "--pickle", action="store_true",
        help="Export graph data as knowledge_graph.pkl",
    )
    ap.add_argument(
        "--debug", action="store_true",
        help="Print each message and its extracted entities",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Only process first N messages (after --offset)",
    )
    ap.add_argument(
        "--offset", type=int, default=0,
        help="Skip first N messages (use with --limit for pagination)",
    )
    return ap


def _build_full_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m full",
        description="Extract + graph in one step.",
    )
    ap.add_argument("json_dir", type=Path, help="Directory containing .json files")
    ap.add_argument(
        "pickle_path", nargs="?", type=Path, default=Path("messages.pkl"),
        help="Intermediate/output pickle path (default: messages.pkl)",
    )
    ap.add_argument(
        "--pickle", action="store_true",
        help="Export graph data as knowledge_graph.pkl",
    )
    ap.add_argument(
        "--debug", action="store_true",
        help="Print each message and its extracted entities",
    )
    ap.add_argument(
        "--limit", type=int, default=0,
        help="Only process first N messages (after --offset)",
    )
    ap.add_argument(
        "--offset", type=int, default=0,
        help="Skip first N messages (use with --limit for pagination)",
    )
    return ap


def _build_join_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m join",
        description="Join multiple conversation files into one normalized JSON.",
    )
    ap.add_argument(
        "-i", "--input", required=True, nargs="+",
        help="Input files/directories (.json, .md)",
    )
    ap.add_argument(
        "-f", "--format", required=True,
        choices=["anthropic", "openai", "gemini"],
        help="Output format",
    )
    ap.add_argument(
        "-o", "--output",
        help="Output file (default: stdout)",
    )
    ap.add_argument(
        "--dedup", action="store_true", default=True,
        help="Deduplicate conversations by content hash (default: on)",
    )
    ap.add_argument(
        "--no-dedup", action="store_false", dest="dedup",
        help="Skip content-hash deduplication",
    )
    return ap


def _build_timeline_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m timeline",
        description="Entity frequency over time.",
    )
    ap.add_argument(
        "graph", nargs="?", type=Path, default=Path("knowledge_graph.pkl"),
        help="Input knowledge graph pickle (default: knowledge_graph.pkl)",
    )
    ap.add_argument(
        "messages", nargs="?", type=Path, default=Path("messages.pkl"),
        help="Messages pickle with timestamps (default: messages.pkl)",
    )
    ap.add_argument(
        "--top", type=int, default=5,
        help="Show top N entities per bucket (default: 5)",
    )
    ap.add_argument(
        "--freq", default="month", choices=["year", "month", "week", "day"],
        help="Time bucket frequency (default: month)",
    )
    ap.add_argument(
        "-o", "--output", type=Path,
        help="Output CSV file (optional)",
    )
    return ap


def _build_export_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m export",
        description="Export knowledge graph to GEXF (Gephi).",
    )
    ap.add_argument(
        "pickle_path", nargs="?", type=Path, default=Path("knowledge_graph.pkl"),
        help="Input knowledge graph pickle (default: knowledge_graph.pkl)",
    )
    ap.add_argument(
        "-o", "--output", type=Path, default=Path("knowledge_graph.gexf"),
        help="Output GEXF file (default: knowledge_graph.gexf)",
    )
    return ap


def _build_split_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m split",
        description="Split a conversations JSON into individual files.",
    )
    ap.add_argument(
        "input_file", nargs="?", default="conversations.json",
        help="Input JSON file (use '-' to read from stdin)",
    )
    ap.add_argument(
        "-o", "--output-dir", default="conversations",
        help="Output directory (default: conversations)",
    )
    return ap


PARSERS = {
    "export": _build_export_parser,
    "extract": _build_extract_parser,
    "timeline": _build_timeline_parser,
    "graph": _build_graph_parser,
    "full": _build_full_parser,
    "join": _build_join_parser,
    "split": _build_split_parser,
}


def main() -> int:
    base = _build_base_parser()
    if len(sys.argv) < 3:
        base.print_help()
        return 1

    mode = sys.argv[2]
    if mode not in PARSERS:
        base.print_help()
        return 1

    remaining = sys.argv[3:]
    args = PARSERS[mode]().parse_args(remaining)

    if mode == "export":
        run_export(args)
    elif mode == "timeline":
        run_timeline(args)
    elif mode == "join":
        run_join(args)
    elif mode == "split":
        run_split(args)
    elif mode == "extract":
        run_extract(args.json_dir, args.pickle_path)
    elif mode == "graph":
        run_graph(
            args.pickle_path,
            export_pickle=args.pickle,
            debug=args.debug,
            limit=args.limit,
            offset=args.offset,
        )
    elif mode == "full":
        run_extract(args.json_dir, args.pickle_path)
        run_graph(
            args.pickle_path,
            export_pickle=args.pickle,
            debug=args.debug,
            limit=args.limit,
            offset=args.offset,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
