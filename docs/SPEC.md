# SPEC.md — convo-tools

## Purpose

Extract, deduplicate, and build knowledge graphs from ChatGPT conversation exports.
Conversations are loaded from JSON files, deduplicated by SHA-256 content hash, and
stored as a pickle. The pickle is then processed with spaCy NER and TF-IDF to produce
a knowledge graph stored as plain Python dicts/sets for compact memory usage.

Also supports joining conversation files from multiple providers (OpenAI, Anthropic,
DeepSeek, Gemini) into a single normalized format, and splitting normalized files back
into individual conversation files.

## Scope

### In scope

- Load ChatGPT conversation JSONs from a directory (the JSON files contain a list of
  conversation objects with `conversation_id` and `mapping` fields).
- Deduplicate messages by SHA-256 hash of the text content.
- Serialize deduplicated messages to a pickle file.
- Build a knowledge graph from the pickle containing:
  - Conversation nodes
  - Message nodes (with role, truncated text)
  - Entity nodes (from spaCy NER: `en_core_web_sm` with `tok2vec` + `ner`)
  - Keyword nodes (from TF-IDF, top 5 keywords per message)
  - `CONTAINS` edges (conversation → message)
  - `REPLIES_TO` edges (parent message → reply message)
  - `MENTIONS` edges (message → entity)
  - `CO_OCCURS_WITH` edges (entity ↔ entity, undirected, deduplicated)
  - `HAS_KEYWORD` edges (message → keyword, with TF-IDF weight)
- Export the graph data structure as a pickle file (via `--pickle` flag).
- Incremental graph updates: load existing `knowledge_graph.pkl`, process only new messages.
- Print per-message entity extractions (via `--debug` flag).
- Limit/paginate messages processed (via `--limit=N` and `--offset=N` flags).
- Report RSS memory at each phase.
- Join conversations from multiple providers/formats into a single normalized JSON.
- Split a conversations JSON into individual conversation files.

### Out of scope

- Web UI or visualization of the graph.
- Graph querying or traversal beyond construction.
- Distributed processing.

## Public API / Interface

### CLI

```
convo-tools -m <mode> [args...]
```

**Modes:**

| Mode | Args | Description |
|------|------|-------------|
| `extract` | `<json_dir> [pickle_path]` | Read JSONs → deduped pickle |
| `graph` | `<pickle_path> [--pickle] [--debug] [--limit N] [--offset N]` | Pickle → knowledge graph |
| `full` | `<json_dir> [pickle_path] [--pickle] [--debug] [--limit N] [--offset N]` | Extract + graph |
| `join` | `-i <path> [-i ...] -f <format> [-o <file>] [--no-dedup]` | Join multi-provider files |
| `split` | `[input_file] [-o <dir>]` | Split conversations JSON |

**Options:**

- `--pickle` — also export the graph data as `knowledge_graph.pkl`
- `--debug` — print each message and its extracted entities
- `--limit N` — only process first N messages

**Default pickle path:** `messages.pkl`

### Python API

```python
from convo_tools import build_graph, extract_messages

# extract: JSON directory → list of deduplicated messages
messages = extract_messages(conversation_dict)

# graph builder: messages → graph dict
graph_data = build_graph(messages, debug=False, limit=0)
# graph_data = {
#     "nodes": dict[str, dict],
#     "edges_contains": set[tuple[str, str]],
#     "edges_replies_to": set[tuple[str, str]],
#     "edges_mentions": set[tuple[str, str]],
#     "edges_cooc": set[tuple[str, str]],
#     "edges_keywords": list[tuple[str, str, float]],
# }
```

## Data Formats

### Input JSON

A ChatGPT conversation export is a JSON file containing a single conversation object
with at least:

```json
{
  "mapping": {
    "<node_id>": {
      "message": {
        "author": {"role": "user"|"assistant"},
        "content": {"content_type": "text", "parts": ["..."]}
      },
      "parent": "<parent_node_id>"
    }
  }
}
```

### Intermediate pickle

Serialized `list[dict]` where each dict has keys:
`id`, `role`, `text`, `parent`, `conversation_id`.

### Graph pickle (via `--pickle`)

Serialized dict with keys:
`nodes`, `edges_contains`, `edges_replies_to`, `edges_mentions`, `edges_cooc`,
`edges_keywords`.

## Edge Cases

1. **Empty conversations directory** — exit gracefully with a message.
1. **Non-text messages** (e.g., code, system, or tool messages) — skip silently.
1. **Duplicate messages** (same SHA-256 hash) — keep only the first occurrence.
1. **Messages longer than 100 000 characters** — split into chunks for spaCy.
1. **Messages with no entities** — no `MENTIONS` edges; no co-occurrence pairs.
1. **TF-IDF produces no keywords** for a message — skip keyword edges for that message.
1. **Orphan messages** (parent not in current graph) — skip `REPLIES_TO` edge.
1. **Same entity pairs co-occurring across multiple messages** — `CO_OCCURS_WITH` edge
   stored once.
1. **Pickle file not found** — crash with `FileNotFoundError`.
1. **Invalid JSON file** in input directory — skip with error message, continue.

## Performance & Constraints

- The entire message list is loaded into memory (expected: 10 000–50 000 messages).
- spaCy model `en_core_web_sm` is loaded with only `tok2vec` + `ner` pipelines.
- TF-IDF uses `max_features=10000`, `ngram_range=(1,2)`, English stop words.
- Co-occurrence edges stored as `set[tuple[str,str]]` to minimize memory.
- No external database; everything is in-memory.
- Target: process 20 000+ messages in under 8 GB RAM.
