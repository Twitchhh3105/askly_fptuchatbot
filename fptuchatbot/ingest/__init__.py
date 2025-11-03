"""
Data ingestion modules for PDF processing, chunking, and indexing.
"""

from fptuchatbot.ingest.bm25 import BM25Index
from fptuchatbot.ingest.chunking import DocumentChunker
from fptuchatbot.ingest.embedder import Embedder
from fptuchatbot.ingest.index_faiss import FAISSIndex
from fptuchatbot.ingest.pdf_loader import PDFLoader
from fptuchatbot.ingest.table_extractor import TableExtractor

__all__ = [
    "PDFLoader",
    "TableExtractor",
    "DocumentChunker",
    "Embedder",
    "FAISSIndex",
    "BM25Index",
]

