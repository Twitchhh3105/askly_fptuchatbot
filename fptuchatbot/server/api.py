"""
FastAPI server for FPT University RAG Chatbot.
"""

import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from fptuchatbot import __version__
from fptuchatbot.ingest.bm25 import BM25Index
from fptuchatbot.ingest.chunking import DocumentChunker
from fptuchatbot.ingest.embedder import Embedder
from fptuchatbot.ingest.index_faiss import FAISSIndex
from fptuchatbot.ingest.pdf_loader import PDFLoader
from fptuchatbot.ingest.table_extractor import TableExtractor
from fptuchatbot.llm.gemini_client import GeminiClient
from fptuchatbot.retrieval.hybrid import HybridRetriever
from fptuchatbot.retrieval.rerank import CrossEncoderReranker
from fptuchatbot.server.schemas import (
    Citation,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
)
from fptuchatbot.utils.config import get_settings
from fptuchatbot.utils.logging import get_logger, setup_logging

# Setup logging
logger = setup_logging("fptuchatbot.server")

# Initialize FastAPI app
app = FastAPI(
    title="FPT University RAG Chatbot API",
    description="Production-ready RAG system for FPT University Q&A",
    version=__version__,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
settings = get_settings()
embedder: Optional[Embedder] = None
bm25_index: Optional[BM25Index] = None
faiss_index: Optional[FAISSIndex] = None
hybrid_retriever: Optional[HybridRetriever] = None
reranker: Optional[CrossEncoderReranker] = None
gemini_client: Optional[GeminiClient] = None


def load_indexes():
    """Load indexes on startup."""
    global embedder, bm25_index, faiss_index, hybrid_retriever, reranker, gemini_client

    logger.info("Loading indexes...")

    try:
        # Load embedder
        embedder = Embedder()

        # Load indexes
        index_dir = settings.get_index_dir()
        bm25_path = index_dir / "bm25_index.pkl"
        faiss_path = index_dir / "faiss_index.faiss"

        if bm25_path.exists() and faiss_path.exists():
            logger.info("Loading BM25 index...")
            bm25_index = BM25Index.from_saved(bm25_path)

            logger.info("Loading FAISS index...")
            faiss_index = FAISSIndex.from_saved(faiss_path)

            # Initialize hybrid retriever
            hybrid_retriever = HybridRetriever(
                bm25_index=bm25_index,
                faiss_index=faiss_index,
                embedder=embedder,
                top_k_bm25=settings.top_k_bm25,
                top_k_faiss=settings.top_k_dense,
            )

            # Initialize reranker
            reranker = CrossEncoderReranker()

            logger.info("Indexes loaded successfully")
        else:
            logger.warning("Indexes not found. Run ingestion first.")

        # Initialize Gemini client
        gemini_client = GeminiClient()

    except Exception as e:
        logger.error(f"Failed to load indexes: {e}")


@app.on_event("startup")
async def startup_event():
    """Run on startup."""
    logger.info("Starting FPT University RAG Chatbot API")
    load_indexes()


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "name": "FPT University RAG Chatbot API",
        "version": __version__,
        "status": "running",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    indexes_loaded = (
        bm25_index is not None
        and faiss_index is not None
        and hybrid_retriever is not None
    )
    gemini_configured = gemini_client is not None and gemini_client.configured

    return HealthResponse(
        status="healthy",
        version=__version__,
        indexes_loaded=indexes_loaded,
        gemini_configured=gemini_configured,
    )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Query endpoint for RAG question answering.
    """
    if not hybrid_retriever:
        raise HTTPException(status_code=503, detail="Indexes not loaded. Run /ingest first.")

    if not gemini_client or not gemini_client.configured:
        raise HTTPException(status_code=503, detail="Gemini not configured. Set GEMINI_API_KEY.")

    start_time = time.time()

    try:
        # Hybrid retrieval
        retrieval_start = time.time()
        results = hybrid_retriever.retrieve(
            query=request.query,
            top_k=settings.top_k_rerank if request.use_rerank else request.top_k,
        )
        retrieval_time = (time.time() - retrieval_start) * 1000

        # Rerank if enabled
        if request.use_rerank and reranker:
            results = reranker.rerank(
                query=request.query,
                results=results,
                top_k=request.top_k,
            )

        # Generate answer
        generation_start = time.time()
        response_data = gemini_client.generate_with_context(
            query=request.query,
            context_chunks=results,
            temperature=request.temperature,
        )
        generation_time = (time.time() - generation_start) * 1000

        total_time = (time.time() - start_time) * 1000

        # Build citations
        citations = [
            Citation(
                source=c["source"],
                page=c["page"],
                score=c["score"],
            )
            for c in response_data["citations"]
        ]

        return QueryResponse(
            answer=response_data["answer"],
            citations=citations,
            num_sources=response_data["num_sources"],
            retrieval_time_ms=retrieval_time,
            generation_time_ms=generation_time,
            total_time_ms=total_time,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search(request: SearchRequest):
    """
    Search endpoint (retrieval only, no LLM generation).
    """
    if not hybrid_retriever:
        raise HTTPException(status_code=503, detail="Indexes not loaded.")

    start_time = time.time()

    try:
        # Hybrid retrieval
        results = hybrid_retriever.retrieve(
            query=request.query,
            top_k=settings.top_k_rerank if request.use_rerank else request.top_k,
        )

        # Rerank if enabled
        if request.use_rerank and reranker:
            results = reranker.rerank(
                query=request.query,
                results=results,
                top_k=request.top_k,
            )

        time_taken = (time.time() - start_time) * 1000

        # Format results
        search_results = [
            SearchResult(
                text=r.get("text", r.get("sentence_chunk", "")),
                source=r.get("source_file", "Unknown"),
                page=r.get("page_number", 0),
                score=r.get("rerank_score") or r.get("hybrid_score", 0),
            )
            for r in results
        ]

        return SearchResponse(
            results=search_results,
            num_results=len(search_results),
            time_taken_ms=time_taken,
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/ingest", response_model=IngestResponse, tags=["Admin"])
async def ingest(request: IngestRequest):
    """
    Ingest documents and build indexes.
    """
    start_time = time.time()

    try:
        data_dir = Path(request.data_dir) if request.data_dir else settings.get_data_dir()
        index_dir = settings.get_index_dir()
        index_dir.mkdir(parents=True, exist_ok=True)

        # Check if indexes exist and force_rebuild is False
        bm25_path = index_dir / "bm25_index.pkl"
        faiss_path = index_dir / "faiss_index.faiss"

        if not request.force_rebuild and bm25_path.exists() and faiss_path.exists():
            return IngestResponse(
                status="skipped",
                num_files=0,
                num_chunks=0,
                time_taken_s=0,
            )

        logger.info(f"Starting ingestion from {data_dir}")

        # Load PDFs
        pdf_loader = PDFLoader()
        pdf_files = sorted(data_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")

        all_pages = []
        for pdf_path in pdf_files:
            pages_data = pdf_loader.load(pdf_path)
            all_pages.extend(pages_data)

        logger.info(f"Loaded {len(all_pages)} pages")

        # Chunk documents
        chunker = DocumentChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            min_tokens=settings.min_chunk_tokens,
            use_semantic=settings.use_semantic_chunking,
        )
        chunks = chunker.chunk_document(all_pages)

        logger.info(f"Created {len(chunks)} chunks")

        # Build embeddings
        global embedder
        if embedder is None:
            embedder = Embedder()

        embeddings = embedder.encode_chunks(chunks)

        # Build FAISS index
        global faiss_index
        faiss_index = FAISSIndex(
            dimension=embedder.embedding_dim,
            index_type=settings.faiss_index_type,
        )
        faiss_index.add(embeddings, chunks)
        faiss_index.save(faiss_path)

        # Build BM25 index
        global bm25_index
        bm25_index = BM25Index()
        bm25_index.build(chunks)
        bm25_index.save(bm25_path)

        # Initialize hybrid retriever
        global hybrid_retriever
        hybrid_retriever = HybridRetriever(
            bm25_index=bm25_index,
            faiss_index=faiss_index,
            embedder=embedder,
        )

        time_taken = time.time() - start_time

        logger.info(f"Ingestion completed in {time_taken:.2f}s")

        return IngestResponse(
            status="success",
            num_files=len(pdf_files),
            num_chunks=len(chunks),
            time_taken_s=time_taken,
        )

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/reload", tags=["Admin"])
async def reload_indexes():
    """Reload indexes from disk."""
    try:
        load_indexes()
        return {"status": "success", "message": "Indexes reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reload failed: {str(e)}")

