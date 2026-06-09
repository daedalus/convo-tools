from __future__ import annotations

import sys
from pathlib import Path

from convo_tools._builder import run_graph
from convo_tools._extract import run_extract


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


def parse_args() -> tuple[str, Path | None, Path, bool, bool, int, int]:
    if len(sys.argv) < 3 or sys.argv[1] != "-m":
        usage()

    mode = sys.argv[2]

    remaining = sys.argv[3:]
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

    return mode, json_dir, pickle_path, export_pickle, debug, limit, offset


def main() -> int:
    mode, json_dir, pickle_path, export_pickle, debug, limit, offset = parse_args()

    if mode in ("extract", "full") and json_dir is not None:
        run_extract(json_dir, pickle_path)

    if mode in ("graph", "full"):
        run_graph(pickle_path, export_pickle, debug, limit, offset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
