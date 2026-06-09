from __future__ import annotations

import argparse
import sys
from pathlib import Path

from convo_tools._builder import run_graph
from convo_tools._extract import run_extract
from convo_tools._join import run_join
from convo_tools._split import run_split


def usage() -> None:
    print(
        f"Usage: {sys.argv[0]} -m <mode> [args...]",
        file=sys.stderr,
    )
    print("  Modes:", file=sys.stderr)
    print(
        "    extract <json_dir> [pickle_path]   — read JSONs → deduped pickle",
        file=sys.stderr,
    )
    print(
        "    graph   <pickle_path> [--pickle] [--debug] [--limit N]  — pickle → knowledge graph",
        file=sys.stderr,
    )
    print(
        "    full    <json_dir> [pickle_path]              — extract + graph",
        file=sys.stderr,
    )
    print(
        "    join    -i <path> [-i ...] -f <format>   — join multi-provider files",
        file=sys.stderr,
    )
    print(
        "    split   [input_file] [-o <dir>]          — split conversations JSON",
        file=sys.stderr,
    )
    print("    Default pickle_path: messages.pkl", file=sys.stderr)
    print(
        "    --pickle  also export graph data as knowledge_graph.pkl", file=sys.stderr
    )
    print(
        "    --debug   print each message and its extracted entities", file=sys.stderr
    )
    print("    --limit N     only process first N messages (after --offset)", file=sys.stderr)
    print(
        "    --offset N    skip first N messages (use with --limit for pagination)",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] != "-m":
        usage()

    mode = sys.argv[2]
    remaining = sys.argv[3:]

    if mode == "join":
        ap = argparse.ArgumentParser(
            description="Join multiple conversation files into one normalized JSON."
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
            "-o", "--output", help="Output file (default: stdout)"
        )
        ap.add_argument(
            "--dedup", action="store_true", default=True,
            help="Deduplicate conversations by content hash (default: on)",
        )
        ap.add_argument(
            "--no-dedup", action="store_false", dest="dedup",
            help="Skip content-hash deduplication",
        )
        run_join(ap.parse_args(remaining))
        return 0

    if mode == "split":
        ap = argparse.ArgumentParser(
            description="Split a conversations JSON into individual files"
        )
        ap.add_argument(
            "input_file", nargs="?", default="conversations.json",
            help="Input JSON file (use '-' to read from stdin)",
        )
        ap.add_argument(
            "-o", "--output-dir", default="conversations",
            help="Output directory (default: conversations)",
        )
        run_split(ap.parse_args(remaining))
        return 0

    if mode not in ("extract", "graph", "full"):
        usage()

    pickle_path = Path("messages.pkl")
    json_dir: Path | None = None
    export_pickle = False
    debug = False
    limit = 0
    offset = 0

    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg == "--pickle":
            export_pickle = True
        elif arg == "--debug":
            debug = True
        elif arg == "--limit":
            i += 1
            if i >= len(remaining):
                usage()
            limit = int(remaining[i])
        elif arg.startswith("--limit="):
            limit = int(arg.split("=", 1)[1])
        elif arg == "--offset":
            i += 1
            if i >= len(remaining):
                usage()
            offset = int(remaining[i])
        elif arg.startswith("--offset="):
            offset = int(arg.split("=", 1)[1])
        elif arg.startswith("--"):
            pass
        elif json_dir is None and mode in ("extract", "full"):
            json_dir = Path(arg)
        elif pickle_path == Path("messages.pkl"):
            pickle_path = Path(arg)
        else:
            usage()
        i += 1

    if mode in ("extract", "full") and json_dir is None:
        usage()

    if mode in ("extract", "full") and json_dir is not None:
        run_extract(json_dir, pickle_path)

    if mode in ("graph", "full"):
        run_graph(pickle_path, export_pickle, debug, limit, offset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
