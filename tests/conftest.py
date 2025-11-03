"""
Pytest configuration and fixtures.
"""

import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "chunk_id": 1,
            "text": "Học phí ngành Công nghệ thông tin là 31.600.000đ/kỳ cho học kỳ 1-3.",
            "page_number": 5,
            "source_file": "hoc_phi.pdf",
            "chunk_token_count": 20,
        },
        {
            "chunk_id": 2,
            "text": "Sinh viên phải đạt điểm trung bình tích lũy tối thiểu 2.0 để tốt nghiệp.",
            "page_number": 10,
            "source_file": "quy_che.pdf",
            "chunk_token_count": 18,
        },
        {
            "chunk_id": 3,
            "text": "Thời gian thực hiện OJT là 4 tháng tại doanh nghiệp.",
            "page_number": 3,
            "source_file": "ojt.pdf",
            "chunk_token_count": 15,
        },
    ]


@pytest.fixture
def sample_pdf_content():
    """Sample PDF text content."""
    return {
        "page_number": 0,
        "text": "Điều 1: Quy định chung. Quy chế này áp dụng cho sinh viên đại học chính quy.",
        "page_char_count": 80,
        "page_word_count": 15,
        "source_file": "test.pdf",
    }


@pytest.fixture
def sample_query():
    """Sample user query."""
    return "Học phí ngành CNTT là bao nhiêu?"

