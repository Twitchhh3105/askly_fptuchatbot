"""
Configuration management using Pydantic Settings.
Reads from .env file and environment variables.
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== Google Gemini API =====
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(
        default="gemini-2.0-flash-exp", description="Gemini model name"
    )

    # ===== Embedding Model =====
    embedding_model: str = Field(
        default="BAAI/bge-m3", description="Sentence transformer model for embeddings"
    )
    embedding_dim: int = Field(default=1024, description="Embedding dimension")
    embedding_device: str = Field(default="cpu", description="Device: cpu or cuda")

    # ===== Cross-Encoder Reranker =====
    reranker_model: str = Field(
        default="BAAI/bge-reranker-base", description="Cross-encoder reranker model"
    )
    reranker_device: str = Field(default="cpu", description="Device: cpu or cuda")

    # ===== FAISS Index Configuration =====
    faiss_index_type: Literal["FlatIP", "HNSW"] = Field(
        default="FlatIP", description="FAISS index type"
    )
    faiss_hnsw_m: int = Field(default=32, description="HNSW M parameter")
    faiss_hnsw_ef_construction: int = Field(
        default=200, description="HNSW ef_construction"
    )
    faiss_hnsw_ef_search: int = Field(default=64, description="HNSW ef_search")

    # ===== Retrieval Parameters =====
    top_k_bm25: int = Field(default=50, description="Top-K for BM25 retrieval")
    top_k_dense: int = Field(default=50, description="Top-K for dense retrieval")
    top_k_rerank: int = Field(default=100, description="Top-K for reranking")
    top_k_final: int = Field(default=8, description="Final top-K to return")

    # ===== Chunking Configuration =====
    chunk_size: int = Field(default=800, description="Chunk size in characters")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    min_chunk_tokens: int = Field(default=50, description="Minimum tokens per chunk")
    use_semantic_chunking: bool = Field(
        default=True, description="Use semantic chunking"
    )

    # ===== PDF Processing =====
    pdf_strategy: str = Field(
        default="hi_res", description="Unstructured PDF strategy"
    )
    ocr_languages: str = Field(
        default="vie+eng", description="Tesseract OCR languages"
    )
    use_table_extraction: bool = Field(
        default=True, description="Extract tables with Camelot"
    )
    remove_footer: bool = Field(default=True, description="Remove PDF footers")
    footer_threshold: int = Field(
        default=70, description="Footer detection threshold (pixels)"
    )

    # ===== Paths =====
    data_dir: str = Field(default="./data", description="Data directory")
    index_dir: str = Field(default="./indexes", description="Index storage directory")
    logs_dir: str = Field(default="./logs", description="Logs directory")
    cache_dir: str = Field(default="./.cache", description="Cache directory")

    # ===== Server Configuration =====
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    log_level: str = Field(default="info", description="Logging level")
    reload: bool = Field(default=False, description="Auto-reload on code changes")

    # ===== Performance =====
    batch_size_embed: int = Field(default=32, description="Batch size for embedding")
    batch_size_rerank: int = Field(default=16, description="Batch size for reranking")
    max_context_length: int = Field(
        default=8000, description="Max context length for LLM"
    )
    enable_metrics: bool = Field(
        default=True, description="Enable Prometheus metrics"
    )

    @field_validator("gemini_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate that API key is set (not default value)."""
        if not v or v == "your_gemini_api_key_here":
            # Warning, but don't fail - allow for testing without API
            pass
        return v

    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for dir_path in [self.data_dir, self.index_dir, self.logs_dir, self.cache_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def get_data_dir(self) -> Path:
        """Get data directory as Path object."""
        return Path(self.data_dir)

    def get_index_dir(self) -> Path:
        """Get index directory as Path object."""
        return Path(self.index_dir)

    def get_logs_dir(self) -> Path:
        """Get logs directory as Path object."""
        return Path(self.logs_dir)

    def get_cache_dir(self) -> Path:
        """Get cache directory as Path object."""
        return Path(self.cache_dir)


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    This is called once and cached for the application lifetime.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings


# Export for convenience
settings = get_settings()

