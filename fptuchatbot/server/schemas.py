"""
Pydantic schemas for API requests and responses.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request schema for query endpoint."""

    query: str = Field(..., description="User query", min_length=1)
    top_k: Optional[int] = Field(8, description="Number of results to return", ge=1, le=50)
    use_rerank: Optional[bool] = Field(True, description="Whether to use reranking")
    temperature: Optional[float] = Field(0.7, description="LLM temperature", ge=0.0, le=2.0)


class Citation(BaseModel):
    """Citation schema."""

    source: str = Field(..., description="Source file name")
    page: str = Field(..., description="Page number")
    score: str = Field(..., description="Relevance score")


class QueryResponse(BaseModel):
    """Response schema for query endpoint."""

    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default_factory=list, description="Source citations")
    num_sources: int = Field(0, description="Number of source chunks used")
    retrieval_time_ms: float = Field(0, description="Retrieval time in milliseconds")
    generation_time_ms: float = Field(0, description="Generation time in milliseconds")
    total_time_ms: float = Field(0, description="Total time in milliseconds")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    indexes_loaded: bool = Field(..., description="Whether indexes are loaded")
    gemini_configured: bool = Field(..., description="Whether Gemini is configured")


class IngestRequest(BaseModel):
    """Request schema for ingest endpoint."""

    data_dir: Optional[str] = Field(None, description="Data directory path")
    force_rebuild: Optional[bool] = Field(False, description="Force rebuild indexes")


class IngestResponse(BaseModel):
    """Response schema for ingest endpoint."""

    status: str = Field(..., description="Ingestion status")
    num_files: int = Field(0, description="Number of files processed")
    num_chunks: int = Field(0, description="Number of chunks created")
    time_taken_s: float = Field(0, description="Time taken in seconds")


class SearchRequest(BaseModel):
    """Request schema for search endpoint."""

    query: str = Field(..., description="Search query")
    top_k: Optional[int] = Field(10, description="Number of results")
    use_rerank: Optional[bool] = Field(True, description="Use reranking")


class SearchResult(BaseModel):
    """Search result schema."""

    text: str = Field(..., description="Chunk text")
    source: str = Field(..., description="Source file")
    page: int = Field(..., description="Page number")
    score: float = Field(..., description="Relevance score")


class SearchResponse(BaseModel):
    """Response schema for search endpoint."""

    results: List[SearchResult] = Field(default_factory=list)
    num_results: int = Field(0)
    time_taken_ms: float = Field(0)

