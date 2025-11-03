"""
Tests for document chunking.
"""

import pytest

from fptuchatbot.ingest.chunking import DocumentChunker


def test_chunker_initialization():
    """Test chunker initialization."""
    chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)

    assert chunker.chunk_size == 500
    assert chunker.chunk_overlap == 100


def test_detect_headers():
    """Test Vietnamese header detection."""
    chunker = DocumentChunker()

    text = """
Chương I: Quy định chung

Điều 1: Phạm vi áp dụng

Khoản 1: Điều khoản này áp dụng cho sinh viên
"""

    headers = chunker.detect_headers(text)

    assert len(headers) >= 2
    assert any("Chương" in h["text"] for h in headers)
    assert any("Điều" in h["text"] for h in headers)


def test_chunk_simple(sample_pdf_content):
    """Test simple chunking."""
    chunker = DocumentChunker(chunk_size=50, use_semantic=False)

    chunks = chunker.chunk_simple(sample_pdf_content["text"])

    assert len(chunks) > 0
    assert all("text" in chunk for chunk in chunks)


def test_chunk_document(sample_pdf_content):
    """Test full document chunking."""
    chunker = DocumentChunker(chunk_size=100, use_semantic=True)

    pages = [sample_pdf_content]
    chunks = chunker.chunk_document(pages)

    assert len(chunks) > 0
    assert all("chunk_id" in chunk for chunk in chunks)
    assert all("page_number" in chunk for chunk in chunks)

