from __future__ import annotations

import csv
import os
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from convo_tools._util import safe_pickle_load

if TYPE_CHECKING:
    import argparse
    from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from convo_tools._graph_db import GraphDB


def _entity_name(db: GraphDB, eid: str) -> str:
    n = db.get_node(eid)
    if n is None:
        return eid.split("::", 2)[-1]
    name = n.get("name")
    return str(name) if name else eid.split("::", 2)[-1]


def _msg_text(db: GraphDB, msg_id: str) -> str:
    n = db.get_node(msg_id)
    if n is None:
        return ""
    t = n.get("text", "")
    return str(t) if t else ""


def _msg_role(db: GraphDB, msg_id: str) -> str:
    n = db.get_node(msg_id)
    if n is None:
        return "?"
    r = n.get("role", "")
    return str(r) if r else "?"


def _search_entities(
    terms: list[str],
    db: GraphDB,
    edges_mentions: list[tuple[str, str]],
) -> tuple[dict[str, float], dict[str, set[str]]]:
    entity_id_to_name: dict[str, str] = {}
    for _msg_id, ent_id in edges_mentions:
        if ent_id not in entity_id_to_name:
            entity_id_to_name[ent_id] = _entity_name(db, ent_id)

    matched_entities: dict[str, float] = {}
    ent_to_msgs: dict[str, set[str]] = defaultdict(set)

    for msg_id, ent_id in edges_mentions:
        ent_to_msgs[ent_id].add(msg_id)

    for eid, name in entity_id_to_name.items():
        name_lower = name.lower()
        score = 0.0
        for term in terms:
            if term in name_lower:
                score += 1.0
            if name_lower.startswith(term) or name_lower.endswith(term):
                score += 0.5
        if score > 0:
            matched_entities[eid] = score

    return matched_entities, ent_to_msgs


def _search_keywords(
    terms: list[str],
    db: GraphDB,
    edges_keywords: list[tuple[str, str, float]],
) -> tuple[dict[str, float], dict[str, set[str]]]:
    kw_id_to_name: dict[str, str] = {}
    for _msg_id, kid, _score in edges_keywords:
        if kid not in kw_id_to_name:
            kw_id_to_name[kid] = _entity_name(db, kid)

    matched_kws: dict[str, float] = {}
    kw_to_msgs: dict[str, set[str]] = defaultdict(set)

    for msg_id, kid, _score in edges_keywords:
        kw_to_msgs[kid].add(msg_id)

    for kid, name in kw_id_to_name.items():
        name_lower = name.lower()
        score = 0.0
        for term in terms:
            if term in name_lower:
                score += 1.0
        if score > 0:
            matched_kws[kid] = score

    return matched_kws, kw_to_msgs


def _build_results(
    query: str,
    db: GraphDB,
    edges_mentions: list[tuple[str, str]],
    edges_keywords: list[tuple[str, str, float]],
    messages_pkl: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    terms = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 1]

    matched_entities, ent_to_msgs = _search_entities(terms, db, edges_mentions)
    matched_kws, kw_to_msgs = _search_keywords(terms, db, edges_keywords)

    entity_msgs: dict[str, set[str]] = defaultdict(set)
    for eid in matched_entities:
        for mid in ent_to_msgs.get(eid, set()):
            entity_msgs[mid].add(eid)

    kw_msgs: dict[str, set[str]] = defaultdict(set)
    for kid in matched_kws:
        for mid in kw_to_msgs.get(kid, set()):
            kw_msgs[mid].add(kid)

    all_candidate_msgs = set(entity_msgs.keys()) | set(kw_msgs.keys())

    if not all_candidate_msgs:
        if messages_pkl:
            txt_matches = _text_search(query, messages_pkl)
            return txt_matches[:20]
        return []

    msg_scores: list[tuple[float, str]] = []
    for mid in all_candidate_msgs:
        score = 0.0
        for eid in entity_msgs.get(mid, set()):
            score += matched_entities.get(eid, 0.0)
        for kid in kw_msgs.get(mid, set()):
            score += matched_kws.get(kid, 0.0)
        msg_scores.append((score, mid))

    msg_scores.sort(key=lambda x: -x[0])

    msg_index: dict[str, dict[str, Any]] = {}
    if messages_pkl:
        for m in messages_pkl:
            msg_index[m["id"]] = m

    results: list[dict[str, Any]] = []
    seen_convos: set[str] = set()

    for score, mid in msg_scores[:50]:
        text = _msg_text(db, mid)
        role = _msg_role(db, mid)
        ents = sorted(entity_msgs.get(mid, set()), key=lambda e: -matched_entities.get(e, 0.0))
        entity_names = [_entity_name(db, e) for e in ents[:5]]
        kws = sorted(kw_msgs.get(mid, set()), key=lambda k: -matched_kws.get(k, 0.0))
        kw_names = [_entity_name(db, k) for k in kws[:5]]

        conv_id = msg_index.get(mid, {}).get("conversation_id", "?") if msg_index else "?"
        ts = msg_index.get(mid, {}).get("create_time") if msg_index else None

        results.append({
            "msg_id": mid,
            "score": score,
            "role": role,
            "text": text[:500],
            "entities": ", ".join(entity_names),
            "keywords": ", ".join(kw_names),
            "conversation_id": conv_id,
            "timestamp": ts,
        })
        seen_convos.add(conv_id)

    return results


def _text_search(
    query: str,
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    texts: list[str] = []
    msg_ids: list[str] = []
    for m in messages:
        t = m.get("text", "")
        if isinstance(t, str) and t.strip():
            texts.append(t)
            msg_ids.append(m["id"])

    if not texts:
        return []

    msg_lookup = {m["id"]: m for m in messages}

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        max_df=0.85,
        min_df=2,
    )
    tfidf = vectorizer.fit_transform(texts)
    q_vec = vectorizer.transform([query])
    sims = cosine_similarity(q_vec, tfidf).flatten()
    top_indices = np.argsort(sims)[::-1][:20]

    results = []
    for idx in top_indices:
        if sims[idx] < 0.05:
            break
        mid = msg_ids[idx]
        m = msg_lookup[mid]
        results.append({
            "msg_id": mid,
            "score": float(sims[idx]),
            "role": str(m.get("role", "?")),
            "text": str(m.get("text", ""))[:500],
            "entities": "",
            "keywords": "",
            "conversation_id": str(m.get("conversation_id", "?")),
            "timestamp": m.get("create_time"),
        })
    return results


def _build_llm_context(
    results: list[dict[str, Any]],
    max_chars: int = 50000,
) -> str:
    conv_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        conv_groups[r["conversation_id"]].append(r)

    parts: list[str] = []
    total_chars = 0

    for conv_id, msgs in conv_groups.items():
        if total_chars >= max_chars:
            break

        conv_part = [f"--- Conversation: {conv_id} ---"]
        for m in msgs:
            line = f"  [{m['role']}] {m['text'][:300]}"
            if m["entities"]:
                line += f"\n    entities: {m['entities']}"
            if m["keywords"]:
                line += f"\n    keywords: {m['keywords']}"
            conv_part.append(line)

        block = "\n".join(conv_part)
        if total_chars + len(block) > max_chars:
            block = block[: max_chars - total_chars]
        parts.append(block)
        total_chars += len(block)

    return "\n\n".join(parts)


def _call_llm(query: str, context: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "Error: ANTHROPIC_API_KEY not set. Set it to use LLM-powered query mode, "
            "or omit --llm for keyword-based search."
        )

    try:
        import anthropic  # noqa: PLC0415
    except ImportError:
        return (
            "Error: anthropic package not installed. "
            "Run: pip install anthropic"
        )

    client = anthropic.Anthropic(api_key=api_key)

    system_prompt = (
        "You are a helpful assistant analyzing a knowledge graph of the user's conversation history. "
        "You have been given relevant messages, entities, and keywords matching their query. "
        "Answer the user's question based on this context. "
        "Cite specific messages or conversations when possible (by conversation ID). "
        "Be concise but thorough. If the context doesn't contain enough information, say so."
    )

    user_prompt = f"The user asks: {query}\n\nRelevant context from conversation history:\n\n{context}"

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = ""
        for block in message.content:
            if hasattr(block, "text"):
                content += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                content += block.get("text", "")
        return content
    except Exception as e:
        return f"Error calling LLM: {e}"


def run_query(db_path: str | Path, args: argparse.Namespace) -> None:
    db = GraphDB(db_path)
    try:
        edges_mentions = db.get_edges_mentions()
        edges_keywords = db.get_edges_keywords()

        import pickle

        messages_pkl: list[dict[str, Any]] | None = None
        try:
            messages_pkl = safe_pickle_load(args.messages)
        except (FileNotFoundError, pickle.UnpicklingError):
            pass

        query = args.query
        if not query:
            print("Error: no query provided.", file=sys.stderr)
            return

        print(f"Query: {query}", flush=True)
        print()

        results = _build_results(query, db, edges_mentions, edges_keywords, messages_pkl)

        if not results:
            print("No matching results found.")
            return

        print(f"Found {len(results)} matching messages across {len({r['conversation_id'] for r in results})} conversations.")
        print()

        if args.llm:
            print("Building LLM context and calling Claude...", flush=True)
            context = _build_llm_context(results, max_chars=args.max_context)
            answer = _call_llm(query, context)
            print()
            print("═══ LLM Response ═══")
            print()
            print(answer)
            print()
        else:
            top_n = min(args.top, len(results))
            print(f"Top {top_n} matching messages:")
            print(f"  {'score':>6s}  {'role':>8s}  {'text':60s}  {'entities':20s}")
            print(f"  {'─'*6}  {'─'*8}  {'─'*60}  {'─'*20}")
            for r in results[:top_n]:
                text = r["text"][:60]
                ents = r["entities"][:20]
                print(f"  {r['score']:>6.3f}  {r['role']:>8s}  {text:60s}  {ents:20s}")

        if args.output:
            with open(args.output, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["msg_id", "score", "role", "text", "entities", "keywords", "conversation_id", "timestamp"])
                for r in results:
                    w.writerow([
                        r["msg_id"], f"{r['score']:.4f}", r["role"],
                        r["text"], r["entities"], r["keywords"],
                        r["conversation_id"], r["timestamp"] or "",
                    ])
            print(f"\nWrote {args.output} ({len(results)} messages)")
    finally:
        db.close()
