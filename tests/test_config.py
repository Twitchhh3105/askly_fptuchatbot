"""
Tests for configuration module.
"""

import pytest

from fptuchatbot.utils.config import Settings, get_settings


def test_settings_defaults():
    """Test default settings values."""
    settings = Settings()

    assert settings.embedding_model == "BAAI/bge-m3"
    assert settings.embedding_dim == 1024
    assert settings.reranker_model == "BAAI/bge-reranker-base"
    assert settings.top_k_final == 8
    assert settings.chunk_size == 800


def test_get_settings():
    """Test settings singleton."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2  # Same instance


def test_settings_directories(temp_dir):
    """Test directory creation."""
    settings = Settings(
        data_dir=str(temp_dir / "data"),
        index_dir=str(temp_dir / "indexes"),
        logs_dir=str(temp_dir / "logs"),
        cache_dir=str(temp_dir / "cache"),
    )

    settings.ensure_directories()

    assert settings.get_data_dir().exists()
    assert settings.get_index_dir().exists()
    assert settings.get_logs_dir().exists()
    assert settings.get_cache_dir().exists()

