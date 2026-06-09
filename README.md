# convo-tools

Build knowledge graphs from ChatGPT conversation exports.

[![Python](https://img.shields.io/pypi/pyversions/convo-tools.svg)](https://pypi.org/project/convo-tools/)

## Install

```bash
pip install convo-tools
```

## Usage

```python
from convo_tools import build_graph, extract_messages

# Extract messages from a ChatGPT conversation JSON
with open("conversation.json") as f:
    conversation = json.load(f)
messages = extract_messages(conversation)

# Build knowledge graph
graph = build_graph(messages)
print(f"Nodes: {len(graph['nodes'])}, Edges: {sum(len(v) for v in graph.values() if isinstance(v, (set, list)))}")
```

## CLI

```bash
# Extract JSON conversations → deduped pickle
convo-tools -m extract /path/to/conversations/ messages.pkl

# Build knowledge graph from pickle
convo-tools -m graph messages.pkl

# Full pipeline
convo-tools -m full /path/to/conversations/ messages.pkl
```

## Development

```bash
git clone https://github.com/daedalus/convo-tools.git
cd convo-tools
pip install -e ".[test]"
pytest
```
