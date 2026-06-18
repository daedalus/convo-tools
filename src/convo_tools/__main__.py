from __future__ import annotations

import argparse
import sys
from pathlib import Path

from convo_tools._builder import run_graph
from convo_tools._centrality import run_centrality
from convo_tools._depth import run_depth
from convo_tools._derived import derive_all
from convo_tools._diff import run_diff
from convo_tools._embed import run_embed
from convo_tools._embeddings import run_embeddings
from convo_tools._export import run_export
from convo_tools._extract import run_extract
from convo_tools._ingest import run_ingest
from convo_tools._join import run_join
from convo_tools._ontology import get_ontology_summary
from convo_tools._query import run_query
from convo_tools._semantic import semantic_search
from convo_tools._serve import run_serve
from convo_tools._similarity import run_similarity
from convo_tools._split import run_split
from convo_tools._temporal import run_temporal
from convo_tools._timeline import run_timeline
from convo_tools._topics import run_topics

_P = Path.home() / ".convo-tools"


def _build_base_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m",
        description="Conversation analysis toolkit.",
    )
    ap.add_argument(
        "-m", "--mode", required=True,
        choices=["extract", "graph", "full", "join", "split", "export", "timeline", "centrality", "similarity", "ingest", "topics", "depth", "diff", "embed", "temporal", "query", "serve", "ontology", "enrich", "derive", "embeddings", "similar", "semantic"],
        help="Operation mode",
    )
    return ap


def _build_extract_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m extract",
        description="Read JSON conversation exports, deduplicate, and save as pickle.",
    )
    ap.add_argument("json_path", type=Path, help="JSON file or directory containing .json files")
    ap.add_argument(
        "pickle_path", nargs="?", type=Path, default=_P / "messages.pkl",
        help="Output pickle path (default: messages.pkl)",
    )
    return ap


def _build_graph_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m graph",
        description="Build a knowledge graph from a pickle of extracted messages.",
    )
    ap.add_argument(
        "pickle_path", nargs="?", type=Path, default=_P / "messages.pkl",
        help="Input pickle path (default: messages.pkl)",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        help="SQLite graph database path (default: knowledge_graph.db)",
    )
    ap.add_argument(
        "--pickle", action="store_true",
        help="Also export graph data as knowledge_graph.pkl (for backward compat)",
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
    ap.add_argument(
        "--only-lang", default="all",
        choices=["all", "en", "es"],
        help="Only process messages in a specific language (default: all configured languages)",
    )
    ap.add_argument(
        "--batch-size", type=int, default=64,
        help="Messages per entity-extraction batch (default: 64, lower = less memory)",
    )
    return ap


def _build_full_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m full",
        description="Full pipeline: extract → graph → enrich → derive → embeddings → export.",
    )
    ap.add_argument("json_path", type=Path, help="JSON file or directory containing .json files")
    ap.add_argument(
        "pickle_path", nargs="?", type=Path, default=_P / "messages.pkl",
        help="Intermediate/output pickle path (default: messages.pkl)",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        help="SQLite graph database path (default: knowledge_graph.db)",
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
    ap.add_argument(
        "--only-lang", default="all",
        choices=["all", "en", "es"],
        help="Only process messages in a specific language",
    )
    ap.add_argument(
        "--batch-size", type=int, default=16,
        help="Messages per entity-extraction batch (default: 16)",
    )
    ap.add_argument(
        "--skip-embeddings", action="store_true",
        help="Skip embedding computation",
    )
    ap.add_argument(
        "--skip-export", action="store_true",
        help="Skip GEXF export",
    )
    ap.add_argument(
        "--embed-dim", type=int, default=32,
        help="Embedding dimension (default: 32)",
    )
    ap.add_argument(
        "--embed-method", default="spectral",
        choices=["spectral", "node2vec"],
        help="Embedding method (default: spectral)",
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


def _build_ingest_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m ingest",
        description="Ingest knowledge graph into Kuzu property graph database.",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        help="Input SQLite graph database (default: knowledge_graph.db)",
    )
    ap.add_argument(
        "--pickle-path", type=Path, default=None,
        dest="pickle_path",
        help="Input knowledge graph pickle (legacy, overrides --db)",
    )
    ap.add_argument(
        "kuzu_path", nargs="?", type=Path, default=_P / "knowledge_graph.kzu",
        help="Output Kuzu database directory (default: knowledge_graph.kzu)",
    )
    ap.add_argument(
        "--overwrite", action="store_true",
        help="Delete existing database directory before ingest",
    )
    return ap


def _build_similarity_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m similarity",
        description="Find similar conversations by entity/keyword Jaccard.",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="Input SQLite graph database (default: knowledge_graph.db)",
    )
    ap.add_argument(
        "messages", nargs="?", type=Path, default=_P / "messages.pkl",
        help="Messages pickle (default: messages.pkl)",
    )
    ap.add_argument(
        "--top", type=int, default=20,
        help="Show top N similar pairs (default: 20)",
    )
    ap.add_argument(
        "--threshold", type=float, default=0.3,
        help="Minimum Jaccard score (default: 0.3)",
    )
    ap.add_argument(
        "--all", dest="all", action="store_true",
        help="Include keywords in similarity (default: entities only)",
    )
    ap.add_argument(
        "-o", "--output", type=Path,
        help="Output CSV file",
    )
    return ap


def _build_centrality_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m centrality",
        description="Find bridge entities via betweenness centrality.",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="Input SQLite graph database (default: knowledge_graph.db)",
    )
    ap.add_argument(
        "--top", type=int, default=20,
        help="Show top N entities (default: 20)",
    )
    ap.add_argument(
        "--samples", type=int, default=0,
        help="Sample size for approximate betweenness "
        "(0=exact, default=min(500, nodes))",
    )
    ap.add_argument(
        "--exact", action="store_true",
        help="Force exact computation (may be slow for large graphs)",
    )
    ap.add_argument(
        "--min-weight", type=int, default=2,
        help="Minimum co-occurrence weight (default: 2; higher = fewer edges, less noise)",
    )
    ap.add_argument(
        "-o", "--output", type=Path,
        help="Output CSV file with all entities",
    )
    return ap


def _build_timeline_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m timeline",
        description="Entity frequency over time.",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="Input SQLite graph database (default: knowledge_graph.db)",
    )
    ap.add_argument(
        "messages", nargs="?", type=Path, default=_P / "messages.pkl",
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
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="Input SQLite graph database (default: knowledge_graph.db)",
    )
    ap.add_argument(
        "-o", "--output", type=Path, default=_P / "knowledge_graph.gexf",
        help="Output GEXF file (default: knowledge_graph.gexf)",
    )
    return ap


def _build_topics_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m topics",
        description="Louvain community detection on entity co-occurrence graph.",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="Input SQLite graph database (default: knowledge_graph.db)",
    )
    ap.add_argument(
        "--top", type=int, default=10,
        help="Show top N entities per cluster (default: 10)",
    )
    ap.add_argument(
        "--min-size", type=int, default=3,
        help="Minimum entities per cluster to display (default: 3)",
    )
    ap.add_argument(
        "--min-weight", type=int, default=2,
        help="Minimum co-occurrence weight (default: 2; higher = fewer edges, less noise)",
    )
    ap.add_argument(
        "-o", "--output", type=Path,
        help="Output CSV file with per-entity cluster assignments",
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


def _build_embed_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m embed",
        description="Compute spectral message embeddings for similarity search.",
    )
    ap.add_argument(
        "pickle_path", nargs="?", type=Path, default=_P / "knowledge_graph.pkl",
        help="Input knowledge graph pickle (default: knowledge_graph.pkl)",
    )
    ap.add_argument(
        "--dim", type=int, default=32,
        help="Embedding dimension (default: 32, capped at min(n_messages, n_entities)-1)",
    )
    ap.add_argument(
        "--similar-to", type=str,
        help="Message ID to find similar messages for",
    )
    ap.add_argument(
        "--top", type=int, default=10,
        help="Number of similar neighbors to show (default: 10)",
    )
    ap.add_argument(
        "--all", action="store_true",
        help="Include keywords in the incidence matrix",
    )
    ap.add_argument(
        "--save", type=Path,
        help="Save embeddings to .npz file",
    )
    ap.add_argument(
        "--load", type=Path,
        help="Load precomputed embeddings from .npz file",
    )
    ap.add_argument(
        "--n-iter", type=int, default=5,
        help="SVD iterations (default: 5, higher = more accurate)",
    )
    ap.add_argument(
        "-o", "--output", type=Path,
        help="Output CSV with nearest neighbors (requires --similar-to)",
    )
    return ap


def _build_diff_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m diff",
        description="Compare two knowledge graphs side-by-side.",
    )
    ap.add_argument("left", type=Path, help="Left (reference) knowledge graph pickle")
    ap.add_argument("right", type=Path, help="Right (comparison) knowledge graph pickle")
    ap.add_argument(
        "--left-label", default="left",
        help="Label for left graph (default: left)",
    )
    ap.add_argument(
        "--right-label", default="right",
        help="Label for right graph (default: right)",
    )
    ap.add_argument(
        "--top", type=int, default=30,
        help="Show top N entities by mention-count diff (default: 30)",
    )
    ap.add_argument(
        "-o", "--output", type=Path,
        help="Output CSV file with per-entity comparison",
    )
    return ap


def _build_depth_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m depth",
        description="Analyze reply-chain depth and branching in conversation DAGs.",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="Input SQLite graph database (default: knowledge_graph.db)",
    )
    ap.add_argument(
        "--top", type=int, default=20,
        help="Show top N conversations by depth (default: 20)",
    )
    ap.add_argument(
        "-o", "--output", type=Path,
        help="Output CSV file with per-conversation metrics",
    )
    return ap


def _build_temporal_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m temporal",
        description="Temporal analysis of entity activity: lifespans, bursts, co-activation.",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="Input SQLite graph database (default: knowledge_graph.db)",
    )
    ap.add_argument(
        "messages", nargs="?", type=Path, default=_P / "messages.pkl",
        help="Messages pickle with timestamps (default: messages.pkl)",
    )
    ap.add_argument(
        "--window", type=int, default=30,
        help="Sliding window size in days (default: 30)",
    )
    ap.add_argument(
        "--top", type=int, default=20,
        help="Show top N entities per metric (default: 20)",
    )
    ap.add_argument(
        "--co-activation", action="store_true",
        help="Show top co-activating entity pairs (same bucket)",
    )
    ap.add_argument(
        "-o", "--output", type=Path,
        help="Output CSV with per-entity temporal metrics",
    )
    return ap


def _build_query_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m query",
        description="Natural language query over conversation history using LLM or keyword search.",
    )
    ap.add_argument(
        "query", type=str,
        help="Natural language query (e.g. 'What did I conclude about HNSW early termination?')",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="Input SQLite graph database (default: knowledge_graph.db)",
    )
    ap.add_argument(
        "messages", nargs="?", type=Path, default=_P / "messages.pkl",
        help="Messages pickle for text and timestamps (default: messages.pkl)",
    )
    ap.add_argument(
        "--llm", action="store_true",
        help="Use LLM for intelligent summarization (requires --api-key or OPENAI_API_KEY)",
    )
    ap.add_argument(
        "--api-key", type=str, default="",
        help="OpenAI API key (default: OPENAI_API_KEY env var)",
    )
    ap.add_argument(
        "--api-base", type=str, default="",
        help="OpenAI-compatible API base URL (for vLLM, Ollama, etc.)",
    )
    ap.add_argument(
        "--model", type=str, default="",
        help="Model name (default: OPENAI_MODEL env var or gpt-4o)",
    )
    ap.add_argument(
        "--top", type=int, default=10,
        help="Number of top results to show (default: 10, keyword mode only)",
    )
    ap.add_argument(
        "--max-context", type=int, default=50000,
        help="Maximum context characters for LLM (default: 50000)",
    )
    ap.add_argument(
        "-o", "--output", type=Path,
        help="Output CSV file with matching messages",
    )
    return ap


def _build_serve_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m serve",
        description="Run the MCP server for LLM-powered knowledge graph querying.",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="Input SQLite graph database (default: knowledge_graph.db)",
    )
    return ap


def _build_ontology_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m ontology",
        description="Print the graph ontology (node types, edge types, attributes).",
    )
    return ap


def _build_enrich_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m enrich",
        description="Enrich graph with semantic attributes (depth, centrality, domains, etc.).",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="SQLite graph database path (default: knowledge_graph.db)",
    )
    return ap


def _build_derive_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m derive",
        description="Derive semantic edges (conversation topics, cross-message links, entity bridges).",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="SQLite graph database path (default: knowledge_graph.db)",
    )
    return ap


def _build_embeddings_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m embeddings",
        description="Compute node embeddings for similarity search.",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="SQLite graph database path (default: knowledge_graph.db)",
    )
    ap.add_argument(
        "--dim", type=int, default=32,
        help="Embedding dimension (default: 32)",
    )
    ap.add_argument(
        "--method", default="spectral", choices=["spectral", "node2vec"],
        help="Embedding method (default: spectral)",
    )
    ap.add_argument(
        "--include-keywords", action="store_true",
        help="Include keywords in embedding computation",
    )
    return ap


def _build_similar_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m similar",
        description="Find similar nodes/conversations using embeddings.",
    )
    ap.add_argument(
        "node_id", type=str,
        help="Node ID to find similar items for",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="SQLite graph database path (default: knowledge_graph.db)",
    )
    ap.add_argument(
        "--top", type=int, default=10,
        help="Number of results (default: 10)",
    )
    ap.add_argument(
        "--min-similarity", type=float, default=0.0,
        help="Minimum cosine similarity threshold (default: 0.0)",
    )
    ap.add_argument(
        "--conversations", action="store_true",
        help="Find similar conversations (default: find similar nodes)",
    )
    return ap


def _build_semantic_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog=f"{sys.argv[0]} -m semantic",
        description="Semantic search combining keywords, derived edges, and embeddings.",
    )
    ap.add_argument(
        "query", type=str,
        help="Natural language query (e.g. 'What did I discuss about Python optimization?')",
    )
    ap.add_argument(
        "--db", type=Path, default=_P / "knowledge_graph.db",
        dest="graph_path",
        help="SQLite graph database path (default: knowledge_graph.db)",
    )
    ap.add_argument(
        "--top", type=int, default=10,
        help="Number of results (default: 10)",
    )
    ap.add_argument(
        "--min-similarity", type=float, default=0.0,
        help="Minimum similarity threshold for embedding matches (default: 0.0)",
    )
    ap.add_argument(
        "--no-embeddings", action="store_true",
        help="Disable embedding similarity",
    )
    ap.add_argument(
        "--no-derived", action="store_true",
        help="Disable derived edge traversal",
    )
    ap.add_argument(
        "--no-keywords", action="store_true",
        help="Disable keyword/TF-IDF matching",
    )
    return ap


PARSERS = {
    "centrality": _build_centrality_parser,
    "depth": _build_depth_parser,
    "derive": _build_derive_parser,
    "diff": _build_diff_parser,
    "embed": _build_embed_parser,
    "embeddings": _build_embeddings_parser,
    "enrich": _build_enrich_parser,
    "ingest": _build_ingest_parser,
    "ontology": _build_ontology_parser,
    "query": _build_query_parser,
    "semantic": _build_semantic_parser,
    "similar": _build_similar_parser,
    "similarity": _build_similarity_parser,
    "temporal": _build_temporal_parser,
    "topics": _build_topics_parser,
    "serve": _build_serve_parser,
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

    if mode == "centrality":
        run_centrality(args.graph_path, args)
    elif mode == "depth":
        run_depth(args.graph_path, args)
    elif mode == "diff":
        run_diff(args)
    elif mode == "embed":
        run_embed(args)
    elif mode == "ingest":
        run_ingest(args)
    elif mode == "query":
        run_query(args.graph_path, args)
    elif mode == "similarity":
        run_similarity(args.graph_path, args)
    elif mode == "temporal":
        run_temporal(args.graph_path, args)
    elif mode == "topics":
        run_topics(args.graph_path, args)
    elif mode == "export":
        run_export(args.graph_path, args)
    elif mode == "timeline":
        run_timeline(args.graph_path, args)
    elif mode == "serve":
        run_serve(
            str(args.graph_path) if args.graph_path else None,
        )
    elif mode == "join":
        run_join(args)
    elif mode == "split":
        run_split(args)
    elif mode == "extract":
        run_extract(args.json_path, args.pickle_path)
    elif mode == "graph":
        run_graph(
            args.pickle_path,
            db_path=args.db,
            export_pickle=args.pickle,
            debug=args.debug,
            limit=args.limit,
            offset=args.offset,
            only_lang=args.only_lang,
            batch_size=args.batch_size,
        )
    elif mode == "full":
        from convo_tools._graph_db import GraphDB
        from convo_tools._derived import derive_all
        from convo_tools._embeddings import run_embeddings
        from convo_tools._export import graph_to_gexf
        import time

        t0 = time.time()
        input_path = Path(args.json_path)
        is_pickle = input_path.suffix in (".pkl", ".pickle")

        if is_pickle:
            print("═══ Step 1/5: Extract (skipped — input is pickle) ═══\n")
            args.pickle_path = input_path
            t1 = t0
        else:
            print("═══ Step 1/5: Extract ═══")
            run_extract(args.json_path, args.pickle_path)
            t1 = time.time()
            print(f"  {t1-t0:.1f}s\n")

        print("═══ Step 2/5: Build Graph ═══")
        run_graph(
            args.pickle_path,
            db_path=args.db,
            export_pickle=args.pickle,
            debug=args.debug,
            limit=args.limit,
            offset=args.offset,
            only_lang=args.only_lang,
            batch_size=args.batch_size,
            enrich=True,
        )
        t2 = time.time()
        print(f"  {t2-t1:.1f}s\n")

        db = GraphDB(args.db)

        print("═══ Step 3/5: Derive Semantic Edges ═══")
        derive_all(db)
        t3 = time.time()
        print(f"  {t3-t2:.1f}s\n")

        if not args.skip_embeddings:
            print("═══ Step 4/5: Compute Embeddings ═══")
            run_embeddings(db, dim=args.embed_dim, method=args.embed_method)
            t4 = time.time()
            print(f"  {t4-t3:.1f}s\n")
        else:
            t4 = t3
            print("═══ Step 4/5: Embeddings (skipped) ═══\n")

        if not args.skip_export:
            print("═══ Step 5/5: Export GEXF ═══")
            gexf_path = args.db.with_suffix(".gexf")
            graph_to_gexf(db, gexf_path)
            t5 = time.time()
            print(f"  {t5-t4:.1f}s\n")
        else:
            t5 = t4
            print("═══ Step 5/5: Export (skipped) ═══\n")

        stats = db.graph_stats()
        emb_count = 0
        try:
            rows = db._conn().execute("SELECT COUNT(*) as c FROM node_embedding").fetchone()
            emb_count = rows["c"] if rows else 0
        except Exception:
            pass

        db.close()

        total = time.time() - t0
        print("═══ Done ═══")
        print(f"  Nodes: {stats['nodes']['total']}")
        print(f"  Edges: {sum(stats['edges'].values())}")
        print(f"  Embeddings: {emb_count}")
        print(f"  Total time: {total:.1f}s")
    elif mode == "ontology":
        print(get_ontology_summary())
    elif mode == "enrich":
        from convo_tools._graph_db import GraphDB
        db = GraphDB(args.graph_path)
        db.enrich_all_semantics()
        db.close()
        print(f"Enriched semantics in {args.graph_path}")
    elif mode == "derive":
        from convo_tools._graph_db import GraphDB
        db = GraphDB(args.graph_path)
        derive_all(db)
        db.close()
        print(f"Derived edges in {args.graph_path}")
    elif mode == "embeddings":
        from convo_tools._graph_db import GraphDB
        db = GraphDB(args.graph_path)
        run_embeddings(
            db,
            dim=args.dim,
            method=args.method,
            include_keywords=args.include_keywords,
        )
        db.close()
        print(f"Computed embeddings in {args.graph_path}")
    elif mode == "similar":
        from convo_tools._graph_db import GraphDB
        db = GraphDB(args.graph_path)
        if args.conversations:
            results = db.find_similar_conversations(args.node_id, top=args.top, min_similarity=args.min_similarity)
            print(f"Conversations similar to {args.node_id} (similarity >= {args.min_similarity}):")
            for cid, score in results:
                print(f"  {cid}: {score:.4f}")
        else:
            results = db.find_similar_nodes(args.node_id, top=args.top, min_similarity=args.min_similarity)
            print(f"Nodes similar to {args.node_id} (similarity >= {args.min_similarity}):")
            for nid, score in results:
                node = db.get_node(nid)
                name = node.get("name", nid) if node else nid
                print(f"  {name} ({nid}): {score:.4f}")
        db.close()
    elif mode == "semantic":
        from convo_tools._graph_db import GraphDB
        db = GraphDB(args.graph_path)
        results = semantic_search(
            db,
            args.query,
            top=args.top,
            use_embeddings=not args.no_embeddings,
            use_derived=not args.no_derived,
            use_keywords=not args.no_keywords,
            min_similarity=args.min_similarity,
        )
        if not results:
            print("No matching results found.")
        else:
            print(f"Found {len(results)} semantic matches for: {args.query}")
            print()
            for i, r in enumerate(results, 1):
                entities = ", ".join(r.get("matched_entities", [])[:3])
                keywords = ", ".join(r.get("matched_keywords", [])[:3])
                print(f"{i}. [{r['source']}] score={r['score']:.3f} role={r['role']}")
                print(f"   {r['text'][:120]}...")
                if entities:
                    print(f"   entities: {entities}")
                if keywords:
                    print(f"   keywords: {keywords}")
                print()
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
