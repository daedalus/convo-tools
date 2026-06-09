from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def run_split(args: argparse.Namespace) -> None:
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.input_file == "-":
        conversations = json.load(sys.stdin)
    else:
        with open(args.input_file, encoding="utf-8") as f:
            conversations = json.load(f)

    for conv in conversations:
        cid = conv.get("conversation_id", conv.get("id"))
        name = conv.get("title", "")
        safe_name = "".join(
            "_" if c.isspace() else c
            for c in name
            if c.isalnum() or c.isspace() or c in ("-", "_")
        ).strip("_")[:50]
        filename = (
            f"{cid}_{safe_name}.json"
            if safe_name
            else f"{cid}.json"
        )
        with open(
            os.path.join(output_dir, filename), "w", encoding="utf-8"
        ) as f:
            json.dump(conv, f, indent=2)

    print(
        f"Split {len(conversations)} conversations into '{output_dir}/'"
    )
