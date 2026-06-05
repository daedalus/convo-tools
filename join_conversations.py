#!/usr/bin/env python3
"""
join_conversations.py — Join multiple conversation files/providers into a
single normalized JSON file.

Supports input from:
  anthropic, deepseek, gemini, openai (json or md), singles

Output formats (--format):
  anthropic, openai, gemini

Usage:
  python join_conversations.py -i <path> [<path> ...] -f <format> [-o <file>]
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime, timezone


def _parse_timestamp(ts) -> float | None:
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return ts / 1000.0 if ts > 1e12 else float(ts)
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S.%f%z",
                    "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(ts.replace("+00:00", "Z").rstrip("Z") + "Z"
                                         if "T" in ts else ts,
                                         fmt).replace(tzinfo=timezone.utc).timestamp()
            except ValueError:
                continue
    return None


def _ts_iso(ts_f: float | None) -> str:
    if ts_f is None:
        ts_f = time.time()
    return datetime.fromtimestamp(ts_f, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _ts_gemini_str(ts_f: float | None) -> str:
    if ts_f is None:
        ts_f = time.time()
    return datetime.fromtimestamp(ts_f, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _ts_gemini_ms(ts_f: float | None) -> int:
    return int((ts_f or time.time()) * 1000)


def _uid(seed: str = "") -> str:
    if seed:
        h = hashlib.md5(seed.encode()).hexdigest()
        return str(uuid.UUID(h[:8] + "-" + h[8:12] + "-" + h[12:16] + "-" + h[16:20] + "-" + h[20:32]))
    return str(uuid.uuid4())


def _sorted_children(mapping, node_id):
    """Get children sorted by their message create_time if available."""
    node = mapping.get(node_id)
    if not node:
        return []
    children = node.get("children", [])
    timed = []
    for cid in children:
        cn = mapping.get(cid, {})
        m = cn.get("message") or {}
        t = m.get("create_time", 0) or 0
        timed.append((t, cid))
    timed.sort(key=lambda x: x[0])
    return [cid for _, cid in timed]


NORMALIZED_CONVERSATION = dict  # {id, title, messages, created_at, updated_at, model, source_format, source_file}
NORMALIZED_MESSAGE = dict  # {role, content, created_at, model}


def _content_hash(conv: dict) -> str:
    payload = [conv.get("title", "")]
    for m in conv.get("messages", []):
        payload.append(m.get("role", ""))
        payload.append(m.get("content", "").strip())
    return hashlib.sha256("\n".join(payload).encode()).hexdigest()


# ── Detect Format ─────────────────────────────────────────────────────────

def detect_format(data, filename: str) -> str:
    if filename.endswith(".md"):
        return "openai_md"
    if not isinstance(data, (dict, list)):
        return "unknown"
    if isinstance(data, dict):
        if "mapping" in data and ("conversation_id" in data or "id" in data):
            return "openai_individual"
        if "uuid" in data and "chat_messages" in data:
            return "anthropic_individual"
        if "exportDate" in data and "conversations" in data:
            return "chatgpt_export_summary"
        return "unknown"
    if not data or not isinstance(data[0], dict):
        return "unknown"
    f = data[0]
    if "conversation_id" in f and "mapping" in f:
        return "openai_master"
    if "uuid" in f and "chat_messages" in f:
        return "anthropic_master"
    if "role" in f and "contents" in f:
        model = f.get("model", "")
        if model == "deepseek":
            return "deepseek"
        if model == "gemini":
            return "gemini"
        if str(f.get("id", "")).startswith("claude_"):
            return "singles_claude"
        if "chatGroupId" in f:
            return "generic_message_array"
    return "unknown"


# ── Parsers ───────────────────────────────────────────────────────────────

def _parse_anthropic(convs: list) -> list[dict]:
    out = []
    for c in convs:
        msgs = []
        for m in c.get("chat_messages", []):
            parts = m.get("content", [])
            text = m.get("text", "") or ""
            if not text:
                text = "\n".join(x.get("text", "") for x in parts if isinstance(x, dict))
            msgs.append({
                "role": "assistant" if m.get("sender") == "assistant" else "user",
                "content": text,
                "created_at": _parse_timestamp(m.get("created_at")),
                "model": None,
            })
        out.append({
            "id": c.get("uuid", _uid()),
            "title": c.get("name", ""),
            "messages": msgs,
            "created_at": _parse_timestamp(c.get("created_at")),
            "updated_at": _parse_timestamp(c.get("updated_at")),
            "model": None,
            "source_format": "anthropic",
            "source_file": "",
        })
    return out


def _parse_openai(items: list) -> list[dict]:
    out = []
    for c in items:
        mapping = c.get("mapping", {})
        if not mapping:
            continue
        root_id = None
        for nid, n in mapping.items():
            if n and isinstance(n, dict) and n.get("parent") is None:
                root_id = nid
                break
        if not root_id:
            continue
        ordered = []
        visited = set()
        stack = [root_id]
        while stack:
            nid = stack.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            node = mapping.get(nid)
            if not node:
                continue
            msg = node.get("message")
            if msg and isinstance(msg, dict):
                role = msg.get("author", {}).get("role", "user")
                if role not in ("user", "assistant"):
                    role = "user"
                parts = msg.get("content", {}).get("parts", [])
                text = "".join(str(p or "") for p in parts).strip()
                if text:
                    ordered.append({
                        "role": role,
                        "content": text,
                        "created_at": _parse_timestamp(msg.get("create_time")),
                        "model": msg.get("metadata", {}).get("model_slug"),
                    })
            for child in _sorted_children(mapping, nid):
                if child not in visited:
                    stack.append(child)
        conv_id = c.get("conversation_id") or c.get("id", _uid())
        out.append({
            "id": conv_id,
            "title": c.get("title", ""),
            "messages": ordered,
            "created_at": _parse_timestamp(c.get("create_time")),
            "updated_at": _parse_timestamp(c.get("update_time")),
            "model": None,
            "source_format": "openai",
            "source_file": "",
        })
    return out


def _parse_message_array(data: list, source_format: str) -> list[dict]:
    groups = {}
    for m in data:
        gid = m.get("chatGroupId", _uid())
        if gid not in groups:
            groups[gid] = {
                "id": gid,
                "title": "",
                "messages": [],
                "created_at": None,
                "updated_at": None,
                "model": m.get("model"),
                "source_format": source_format,
                "source_file": "",
            }
        role = m.get("role", "user")
        contents = m.get("contents", [])
        text_parts = []
        for c in contents:
            if isinstance(c, dict):
                t = c.get("type", "")
                if t in ("text", "markdown", "thinking"):
                    text_parts.append(c.get("content", ""))
                elif t == "attachment":
                    aname = c.get("attachment", {}).get("name", "")
                    if aname:
                        text_parts.append(f"[Attachment: {aname}]")
        full = "\n".join(text_parts).strip()
        if not full:
            continue
        groups[gid]["messages"].append({
            "role": role,
            "content": full,
            "created_at": _parse_timestamp(m.get("created_at")),
            "model": m.get("model") or m.get("displayModel"),
        })
        ts = _parse_timestamp(m.get("created_at"))
        if ts:
            g = groups[gid]
            if g["created_at"] is None or ts < g["created_at"]:
                g["created_at"] = ts
            if g["updated_at"] is None or ts > g["updated_at"]:
                g["updated_at"] = ts
    result = list(groups.values())
    for conv in result:
        for msg in conv["messages"]:
            if msg["role"] == "user" and msg["content"]:
                conv["title"] = msg["content"][:80].strip()
                break
    return result


def _parse_openai_md(text: str, source_file: str = "") -> list[dict]:
    lines = text.split("\n")
    title = ""
    msgs = []
    cur_role = None
    cur_lines = []
    role_map = {
        "you": "user", "chatgpt": "assistant",
        "human": "user", "assistant": "assistant",
        "user": "user",
    }
    for line in lines:
        tm = re.match(r'^#\s+(.+)$', line)
        if tm:
            title = tm.group(1).strip()
            continue
        rm = re.match(r'^####\s+(You|ChatGPT|Human|Assistant|User):\s*$', line, re.IGNORECASE)
        if rm:
            if cur_role and cur_lines:
                content = "\n".join(cur_lines).strip()
                if content:
                    msgs.append({"role": cur_role, "content": content, "created_at": None, "model": None})
            cur_role = role_map.get(rm.group(1).lower(), "user")
            cur_lines = []
            continue
        if cur_role:
            cur_lines.append(line)
    if cur_role and cur_lines:
        content = "\n".join(cur_lines).strip()
        if content:
            msgs.append({"role": cur_role, "content": content, "created_at": None, "model": None})
    if not title and msgs:
        title = msgs[0]["content"][:80] if msgs[0]["content"] else ""
    return [{
        "id": _uid(source_file),
        "title": title,
        "messages": msgs,
        "created_at": None,
        "updated_at": None,
        "model": None,
        "source_format": "openai_md",
        "source_file": source_file,
    }]


PARSER_DISPATCH = {
    "anthropic_master": _parse_anthropic,
    "anthropic_individual": lambda d: _parse_anthropic([d]),
    "openai_master": _parse_openai,
    "openai_individual": lambda d: _parse_openai([d]),
    "deepseek": lambda d: _parse_message_array(d, "deepseek"),
    "gemini": lambda d: _parse_message_array(d, "gemini"),
    "singles_claude": lambda d: _parse_message_array(d, "singles_claude"),
    "generic_message_array": lambda d: _parse_message_array(d, "generic"),
}


def parse_file(filepath: str) -> list[dict]:
    filepath = str(filepath)
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".md":
        with open(filepath, encoding="utf-8") as f:
            return _parse_openai_md(f.read(), filepath)
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)
    fmt = detect_format(data, filepath)
    if fmt == "chatgpt_export_summary":
        return []
    parser = PARSER_DISPATCH.get(fmt)
    if not parser:
        print(f"  Skipping {filepath}: unknown format ({fmt})", file=sys.stderr)
        return []
    convs = parser(data)
    for c in convs:
        c["source_file"] = filepath
    return convs


def collect_files(paths: list[str]) -> list[str]:
    files = []
    for p in paths:
        if os.path.isfile(p):
            if os.path.splitext(p)[1].lower() in (".json", ".md"):
                files.append(p)
        elif os.path.isdir(p):
            for root, _dirs, fnames in os.walk(p):
                for fn in fnames:
                    if os.path.splitext(fn)[1].lower() in (".json", ".md"):
                        files.append(os.path.join(root, fn))
    return sorted(set(files))


# ── Writers ───────────────────────────────────────────────────────────────

def write_anthropic(conversations: list[dict]) -> list[dict]:
    out = []
    for conv in conversations:
        cmsgs = []
        for i, m in enumerate(conv["messages"]):
            muid = _uid(f"{conv['id']}_{i}")
            ts = _ts_iso(m["created_at"])
            cmsgs.append({
                "uuid": muid,
                "text": m["content"],
                "content": [{"type": "text", "text": m["content"]}],
                "sender": "human" if m["role"] == "user" else m["role"],
                "created_at": ts,
                "updated_at": ts,
                "attachments": [],
                "files": [],
            })
        ca = conv.get("created_at")
        ua = conv.get("updated_at") or ca
        out.append({
            "uuid": conv["id"],
            "name": conv["title"],
            "summary": "",
            "created_at": _ts_iso(ca),
            "updated_at": _ts_iso(ua),
            "account": {"uuid": "00000000-0000-0000-0000-000000000000"},
            "chat_messages": cmsgs,
        })
    return out


def write_openai(conversations: list[dict]) -> list[dict]:
    out = []
    for conv in conversations:
        mapping = {}
        root_id = _uid(f"{conv['id']}_root")
        mapping[root_id] = {"id": root_id, "message": None, "parent": None, "children": []}
        prev_id = root_id
        for i, m in enumerate(conv["messages"]):
            mid = _uid(f"{conv['id']}_{i}")
            ts = m["created_at"] or time.time()
            mapping[mid] = {
                "id": mid,
                "message": {
                    "id": mid,
                    "author": {"role": m["role"], "name": None, "metadata": {}},
                    "create_time": ts,
                    "update_time": None,
                    "content": {"content_type": "text", "parts": [m["content"]]},
                    "status": "finished_successfully",
                    "end_turn": True,
                    "weight": 1.0,
                    "metadata": {"model_slug": m.get("model") or "auto"},
                    "recipient": "all",
                },
                "parent": prev_id,
                "children": [],
            }
            if prev_id in mapping and mapping[prev_id] is not None:
                pnode = mapping[prev_id]
                if pnode.get("message") is not None or prev_id == root_id:
                    pnode.setdefault("children", []).append(mid)
            prev_id = mid
        cid = conv["id"]
        out.append({
            "conversation_id": cid,
            "id": cid,
            "title": conv["title"],
            "create_time": conv["created_at"] or time.time(),
            "update_time": conv.get("updated_at") or conv["created_at"] or time.time(),
            "current_node": prev_id,
            "default_model_slug": "auto",
            "mapping": mapping,
        })
    return out


def write_gemini(conversations: list[dict]) -> list[dict]:
    out = []
    for conv in conversations:
        for i, m in enumerate(conv["messages"]):
            h = hashlib.md5(f"{conv['id']}_{i}".encode()).hexdigest()[:16]
            rid = "assistant" if m["role"] == "assistant" else "user"
            ts_f = m["created_at"] or time.time()
            out.append({
                "id": f"r_{h}_{rid}",
                "chatGroupId": conv["id"],
                "role": rid,
                "model": "gemini",
                "displayModel": "Gemini",
                "contents": [{"type": "text", "content": m["content"]}],
                "created_at": _ts_gemini_str(ts_f),
                "updated_at": _ts_gemini_ms(ts_f),
            })
    return out


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Join multiple conversation files into one normalized JSON file."
    )
    ap.add_argument("-i", "--input", required=True, nargs="+",
                    help="Input files/directories (.json, .md)")
    ap.add_argument("-f", "--format", required=True,
                    choices=["anthropic", "openai", "gemini"],
                    help="Output format")
    ap.add_argument("-o", "--output", help="Output file (default: stdout)")
    ap.add_argument("--dedup", action="store_true", default=True,
                    help="Deduplicate conversations by content hash (default: on)")
    ap.add_argument("--no-dedup", action="store_false", dest="dedup",
                    help="Skip content-hash deduplication")
    args = ap.parse_args()

    files = collect_files(args.input)
    if not files:
        print("No conversation files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} file(s). Processing...", file=sys.stderr)

    all_convs = []
    for f in files:
        try:
            convs = parse_file(f)
            all_convs.extend(convs)
            print(f"  {f} → {len(convs)} conversation(s)", file=sys.stderr)
        except Exception as e:
            print(f"  Error: {f}: {e}", file=sys.stderr)

    print(f"\nTotal: {len(all_convs)} conversation(s) from {len(files)} file(s).", file=sys.stderr)

    if not all_convs:
        print("Nothing to write.", file=sys.stderr)
        sys.exit(1)

    if args.dedup:
        seen = set()
        unique = []
        dup_count = 0
        for conv in all_convs:
            h = _content_hash(conv)
            if h in seen:
                dup_count += 1
            else:
                seen.add(h)
                unique.append(conv)
        if dup_count:
            print(f"Dedup: removed {dup_count} duplicate(s), {len(unique)} unique remain.",
                  file=sys.stderr)
        all_convs = unique

    writers = {"anthropic": write_anthropic, "openai": write_openai, "gemini": write_gemini}
    out = writers[args.format](all_convs)

    blob = json.dumps(out, indent=2, ensure_ascii=False)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(blob)
        print(f"Wrote {len(out)} items to {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(blob)
        sys.stdout.write("\n")


if __name__ == "__main__":
    main()
