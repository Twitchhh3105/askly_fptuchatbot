"""
End-to-end integration tests.
"""

import pytest


@pytest.mark.integration
def test_full_rag_pipeline():
    """Test complete RAG pipeline (requires indexes)."""
    pytest.skip("Integration test - requires full system setup with indexes")


@pytest.mark.integration  
def test_api_query_endpoint():
    """Test /query API endpoint."""
    pytest.skip("Integration test - requires running server")


@pytest.mark.integration
def test_api_ingest_endpoint():
    """Test /ingest API endpoint."""
    pytest.skip("Integration test - requires running server")

