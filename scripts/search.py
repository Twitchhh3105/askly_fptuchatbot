#!/usr/bin/env python3
"""
Search script for testing retrieval without LLM.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fptuchatbot.ingest.bm25 import BM25Index
from fptuchatbot.ingest.embedder import Embedder
from fptuchatbot.ingest.index_faiss import FAISSIndex
from fptuchatbot.retrieval.hybrid import HybridRetriever
from fptuchatbot.retrieval.rerank import CrossEncoderReranker
from fptuchatbot.utils.config import get_settings
from fptuchatbot.utils.logging import setup_logging

logger = setup_logging("search")


def main():
    """Main search function."""
    parser = argparse.ArgumentParser(description="Search RAG system")
    parser.add_argument(
        "-q", "--query",
        type=str,
        required=True,
        help="Search query",
    )
    parser.add_argument(
        "-k", "--top-k",
        type=int,
        default=8,
        help="Number of results to return",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Skip reranking",
    )
    parser.add_argument(
        "--show-scores",
        action="store_true",
        help="Show detailed scores",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Index directory",
    )

    args = parser.parse_args()

    settings = get_settings()
    index_dir = Path(args.index_dir) if args.index_dir else settings.get_index_dir()

    # Load indexes
    logger.info("Loading indexes...")
    bm25_path = index_dir / "bm25_index.pkl"
    faiss_path = index_dir / "faiss_index.faiss"

    if not bm25_path.exists() or not faiss_path.exists():
        logger.error("Indexes not found. Run ingestion first.")
        sys.exit(1)

    bm25_index = BM25Index.from_saved(bm25_path)
    faiss_index = FAISSIndex.from_saved(faiss_path)
    embedder = Embedder()

    # Initialize retriever
    retriever = HybridRetriever(
        bm25_index=bm25_index,
        faiss_index=faiss_index,
        embedder=embedder,
    )

    # Search
    logger.info(f"Query: {args.query}")
    results = retriever.retrieve(args.query, top_k=settings.top_k_rerank)

    # Rerank if enabled
    if not args.no_rerank:
        logger.info("Reranking results...")
        reranker = CrossEncoderReranker()
        results = reranker.rerank(args.query, results, top_k=args.top_k)

    # Display results
    print(f"\n{'='*80}")
    print(f"Query: {args.query}")
    print(f"Found {len(results)} results")
    print(f"{'='*80}\n")

    for i, result in enumerate(results, 1):
        text = result.get("text", result.get("sentence_chunk", ""))
        source = result.get("source_file", "Unknown")
        page = result.get("page_number", "?")

        print(f"[{i}] {source} (Page {page})")

        if args.show_scores:
            bm25_score = result.get("bm25_score", 0)
            faiss_score = result.get("faiss_score", 0)
            hybrid_score = result.get("hybrid_score", 0)
            rerank_score = result.get("rerank_score", 0)

            print(f"    BM25: {bm25_score:.3f} | FAISS: {faiss_score:.3f} | "
                  f"Hybrid: {hybrid_score:.3f} | Rerank: {rerank_score:.3f}")

        print(f"    {text[:200]}...")
        print()


if __name__ == "__main__":
    main()

