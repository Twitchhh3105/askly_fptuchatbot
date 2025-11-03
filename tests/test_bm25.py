"""
Tests for BM25 index.
"""

import pytest

from fptuchatbot.ingest.bm25 import BM25Index


def test_bm25_initialization():
    """Test BM25 initialization."""
    bm25 = BM25Index(tokenizer="simple")

    assert bm25.tokenizer_type == "simple"
    assert bm25.bm25 is None


def test_bm25_tokenize():
    """Test BM25 tokenization."""
    bm25 = BM25Index()

    tokens = bm25.tokenize("Học phí ngành CNTT là bao nhiêu?")

    assert len(tokens) > 0
    assert all(isinstance(t, str) for t in tokens)


def test_bm25_build_and_search(sample_chunks):
    """Test BM25 build and search."""
    bm25 = BM25Index()
    bm25.build(sample_chunks)

    assert bm25.bm25 is not None
    assert len(bm25.chunks) == len(sample_chunks)

    # Search
    results = bm25.search("học phí", top_k=2)

    assert len(results) <= 2
    assert all("bm25_score" in r for r in results)


def test_bm25_save_load(sample_chunks, temp_dir):
    """Test BM25 save and load."""
    bm25 = BM25Index()
    bm25.build(sample_chunks)

    # Save
    save_path = temp_dir / "bm25_test.pkl"
    bm25.save(save_path)

    assert save_path.exists()

    # Load
    bm25_loaded = BM25Index.from_saved(save_path)

    assert len(bm25_loaded.chunks) == len(sample_chunks)
    assert bm25_loaded.bm25 is not None

