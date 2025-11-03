#!/usr/bin/env python3
"""
Ingestion script for building indexes from PDF documents.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fptuchatbot.ingest.bm25 import BM25Index
from fptuchatbot.ingest.chunking import DocumentChunker
from fptuchatbot.ingest.data_audit import DataAuditor
from fptuchatbot.ingest.embedder import Embedder
from fptuchatbot.ingest.index_faiss import FAISSIndex
from fptuchatbot.ingest.pdf_loader import PDFLoader
from fptuchatbot.utils.config import get_settings
from fptuchatbot.utils.logging import setup_logging
from fptuchatbot.utils.timing import Timer
from fptuchatbot.ingest.table_extractor import TableExtractor
logger = setup_logging("ingest")


def main():
    """Main ingestion function."""
    parser = argparse.ArgumentParser(description="Ingest PDFs and build indexes")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory containing PDFs",
    )
    parser.add_argument(
        "--index-dir",
        type=str,
        default=None,
        help="Output directory for indexes",
    )
    parser.add_argument(
        "--run-audit",
        action="store_true",
        help="Run data audit before ingestion",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only run audit, skip ingestion",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if indexes exist",
    )

    args = parser.parse_args()

    settings = get_settings()
    data_dir = Path(args.data_dir) if args.data_dir else settings.get_data_dir()
    index_dir = Path(args.index_dir) if args.index_dir else settings.get_index_dir()

    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Index directory: {index_dir}")

    # Run audit if requested
    if args.run_audit or args.audit_only:
        logger.info("Running data audit...")
        auditor = DataAuditor(data_dir)
        report = auditor.run_full_audit(output_path=index_dir / "data_report.json")

        logger.info(f"\n===== Data Audit Summary =====")
        logger.info(f"Total files: {report['summary']['total_files']}")
        logger.info(f"Total pages: {report['summary']['total_pages']}")
        logger.info(f"Parse success rate: {report['summary']['parse_success_rate']}%")

        if args.audit_only:
            logger.info("Audit complete. Exiting.")
            return

    # Check if indexes exist
    bm25_path = index_dir / "bm25_index.pkl"
    faiss_path = index_dir / "faiss_index.faiss"

    if not args.force and bm25_path.exists() and faiss_path.exists():
        logger.warning("Indexes already exist. Use --force to rebuild.")
        return

    # Start ingestion
    logger.info("Starting ingestion...")

    with Timer("Total ingestion"):
        # 1. Load PDFs
        logger.info("Loading PDFs...")
        pdf_loader = PDFLoader(
            strategy=settings.pdf_strategy,
            ocr_languages=settings.ocr_languages,
            remove_footer=settings.remove_footer,
        )

        pdf_files = sorted(data_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")

        all_pages = []
        for pdf_path in pdf_files:
            logger.info(f"Processing {pdf_path.name}...")
            pages_data = pdf_loader.load(pdf_path)
            all_pages.extend(pages_data)

        logger.info(f"Loaded {len(all_pages)} pages total")

        # 2. Extract tables (if enabled)
        if settings.use_table_extraction:
            logger.info("Extracting tables from PDFs...")
            table_extractor = TableExtractor()
            for pdf_path in pdf_files:
                logger.info(f"Extracting tables from {pdf_path.name}...")
                tables = table_extractor.extract_tables_with_fallback(pdf_path)
                if tables:
                    logger.info(f"Found {len(tables)} tables in {pdf_path.name}")
                    # Merge tables with pages_data
                    all_pages = table_extractor.merge_tables_with_text(all_pages, tables)
            logger.info("Table extraction complete")

        # 3. Chunk documents
        logger.info("Chunking documents...")
        chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_tokens=settings.min_chunk_tokens,
            use_semantic=settings.use_semantic_chunking,
        )
        chunks = chunker.chunk_document(all_pages)
        logger.info(f"Created {len(chunks)} chunks")

        # Save chunks for reference
        import json
        chunks_path = index_dir / "chunks.json"
        index_dir.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks[:100], f, ensure_ascii=False, indent=2)  # Sample
        logger.info(f"Saved sample chunks to {chunks_path}")

        # 4. Generate embeddings
        logger.info("Generating embeddings...")
        embedder = Embedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.batch_size_embed,
        )

        embeddings = embedder.encode_chunks(chunks, text_field="text")
        logger.info(f"Generated embeddings: shape {embeddings.shape}")

        # 5. Build FAISS index
        logger.info("Building FAISS index...")
        faiss_index = FAISSIndex(
            dimension=embedder.embedding_dim,
            index_type=settings.faiss_index_type,
        )
        faiss_index.add(embeddings, chunks)
        faiss_index.save(faiss_path)
        logger.info(f"FAISS index saved to {faiss_path}")

        # 6. Build BM25 index
        logger.info("Building BM25 index...")
        bm25_index = BM25Index(tokenizer="underthesea", use_stopwords=True)  # Vietnamese tokenizer
        bm25_index.build(chunks, text_field="text")
        bm25_index.save(bm25_path)
        logger.info(f"BM25 index saved to {bm25_path}")

        # 6. Save stats
        stats = {
            "num_files": len(pdf_files),
            "num_pages": len(all_pages),
            "num_chunks": len(chunks),
            "embedding_dim": embedder.embedding_dim,
            "embedding_model": embedder.model_name,
            "faiss_index_type": settings.faiss_index_type,
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
        }

        stats_path = index_dir / "index_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Index stats saved to {stats_path}")

    logger.info("Ingestion complete!")
    logger.info(f"Indexes saved to: {index_dir}")


if __name__ == "__main__":
    main()

