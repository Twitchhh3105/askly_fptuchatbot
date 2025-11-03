"""
FastAPI server for RAG chatbot.
"""

from fptuchatbot.server.api import app
from fptuchatbot.server.schemas import QueryRequest, QueryResponse

__all__ = ["app", "QueryRequest", "QueryResponse"]

