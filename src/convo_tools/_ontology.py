from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NodeSemantics:
    label: str
    description: str
    attributes: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class EdgeSemantics:
    name: str
    domain: str
    range: str
    description: str
    symmetric: bool = False
    weighted: bool = False
    cardinality: str = "N:N"


NODE_TYPES: dict[str, NodeSemantics] = {
    "Conversation": NodeSemantics(
        label="Conversation",
        description="A single chat thread with a provider (OpenAI, Anthropic, etc.)",
        attributes={
            "topic_summary": "LLM-generated summary of the conversation topic",
            "dominant_entities": "Comma-separated top entities by mention count",
            "depth_metrics": "JSON with max_depth, mean_depth, branching_factor",
        },
    ),
    "Message": NodeSemantics(
        label="Message",
        description="A single utterance (user or assistant) in a conversation",
        attributes={
            "role": "Speaker role: 'user' or 'assistant'",
            "text": "Truncated message text (max 1000 chars)",
            "depth_in_chain": "Integer depth in the reply DAG (0 = root)",
            "centrality_score": "Betweenness centrality in the reply graph",
        },
    ),
    "Entity": NodeSemantics(
        label="Entity",
        description="A named entity extracted via spaCy NER",
        attributes={
            "name": "Normalized entity text (lowercase)",
            "entity_type": "NER label: PERSON, ORG, GPE, etc.",
            "mention_count": "Total messages mentioning this entity",
            "domain": "Inferred topic domain from co-occurrence cluster",
        },
    ),
    "Keyword": NodeSemantics(
        label="Keyword",
        description="A TF-IDF-significant term from message text",
        attributes={
            "name": "The keyword/phrase",
            "avg_tfidf": "Mean TF-IDF weight across messages",
            "document_frequency": "Number of messages containing this keyword",
        },
    ),
}

EDGE_TYPES: dict[str, EdgeSemantics] = {
    "CONTAINS": EdgeSemantics(
        name="CONTAINS",
        domain="Conversation",
        range="Message",
        description="A conversation contains a message",
        cardinality="1:N",
    ),
    "REPLIES_TO": EdgeSemantics(
        name="REPLIES_TO",
        domain="Message",
        range="Message",
        description="A message is a reply to another message (directed tree/DAG)",
        cardinality="1:1",
    ),
    "MENTIONS": EdgeSemantics(
        name="MENTIONS",
        domain="Message",
        range="Entity",
        description="A message mentions a named entity",
        cardinality="N:N",
    ),
    "CO_OCCURS_WITH": EdgeSemantics(
        name="CO_OCCURS_WITH",
        domain="Entity",
        range="Entity",
        description="Two entities co-occur in the same message",
        symmetric=True,
        weighted=True,
        cardinality="N:N",
    ),
    "HAS_KEYWORD": EdgeSemantics(
        name="HAS_KEYWORD",
        domain="Message",
        range="Keyword",
        description="A message has a TF-IDF keyword",
        weighted=True,
        cardinality="N:N",
    ),
    "CONVERSATION_TOPIC": EdgeSemantics(
        name="CONVERSATION_TOPIC",
        domain="Conversation",
        range="Entity",
        description="An entity is a dominant topic of a conversation (appears in >30% of messages)",
        weighted=True,
        cardinality="N:N",
    ),
    "CROSS_MESSAGE_LINK": EdgeSemantics(
        name="CROSS_MESSAGE_LINK",
        domain="Message",
        range="Message",
        description="Two messages share significant entity/keyword overlap (Jaccard > 0.3)",
        weighted=True,
        cardinality="N:N",
    ),
    "ENTITY_BRIDGE": EdgeSemantics(
        name="ENTITY_BRIDGE",
        domain="Entity",
        range="Entity",
        description="Two entities are connected through high-centrality bridge entities",
        weighted=True,
        cardinality="N:N",
    ),
}


def get_ontology_summary() -> str:
    lines = ["# Graph Ontology\n"]

    lines.append("## Node Types\n")
    for ns in NODE_TYPES.values():
        lines.append(f"### {ns.label}")
        lines.append(f"{ns.description}\n")
        if ns.attributes:
            lines.append("| Attribute | Description |")
            lines.append("|-----------|-------------|")
            for attr, desc in ns.attributes.items():
                lines.append(f"| `{attr}` | {desc} |")
            lines.append("")

    lines.append("## Edge Types\n")
    for es in EDGE_TYPES.values():
        sym = " (symmetric)" if es.symmetric else ""
        wtd = " (weighted)" if es.weighted else ""
        lines.append(
            f"- **{es.name}**: {es.domain} → {es.range}{sym}{wtd} — {es.description}"
        )

    return "\n".join(lines)


def validate_node_attrs(label: str, attrs: dict[str, Any]) -> list[str]:
    sem = NODE_TYPES.get(label)
    if sem is None:
        return [f"Unknown node label: {label}"]
    errors = []
    known = {"label", "role", "text", "name", "entity_type", "create_time"}
    for key in attrs:
        if key not in known and key not in sem.attributes:
            errors.append(f"Unexpected attribute '{key}' for node type '{label}'")
    return errors
