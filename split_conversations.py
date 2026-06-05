#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser(description="Split a conversations JSON into individual files")
parser.add_argument("input_file", nargs="?", default="conversations.json",
                    help="Input JSON file (use '-' to read from stdin)")
parser.add_argument("-o", "--output-dir", default="conversations",
                    help="Output directory (default: conversations)")
args = parser.parse_args()

output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

if args.input_file == "-":
    conversations = json.load(sys.stdin)
else:
    with open(args.input_file, "r") as f:
        conversations = json.load(f)

for conv in conversations:
    cid = conv["conversation_id"]
    name = conv.get("title", "")
    safe_name = "".join("_" if c.isspace() else c for c in name if c.isalnum() or c.isspace() or c in ("-", "_")).strip("_")[:50]
    filename = f"{cid}_{safe_name}.json" if safe_name else f"{cid}.json"

    with open(os.path.join(output_dir, filename), "w") as f:
        json.dump(conv, f, indent=2)

print(f"Split {len(conversations)} conversations into '{output_dir}/'")
