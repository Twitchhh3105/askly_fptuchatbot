"""
FPT University RAG Chatbot - Production-ready system with Hybrid Retrieval.

This package provides a complete RAG pipeline for Vietnamese Q&A with:
- PDF processing (Unstructured + Camelot for tables)
- Hybrid retrieval (BM25 + FAISS)
- Cross-encoder reranking
- Google Gemini LLM integration
- FastAPI server

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "FPT University AI Team"

from fptuchatbot.utils.config import Settings

__all__ = ["Settings", "__version__"]

