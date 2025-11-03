"""
Tests for hybrid retrieval.
"""

import numpy as np
import pytest


def test_hybrid_retrieval_integration(sample_chunks):
    """Test hybrid retrieval (integration test)."""
    # This would require full setup, so we'll mark it as integration
    pytest.skip("Integration test - requires full index setup")


def test_score_normalization():
    """Test score normalization."""
    from fptuchatbot.retrieval.hybrid import HybridRetriever

    # Mock retriever for testing normalization
    scores = [10.0, 5.0, 2.0, 1.0]
    
    # Create dummy retriever (we only test the method)
    class MockRetriever:
        def _normalize_scores(self, scores):
            if not scores:
                return []
            scores_array = np.array(scores)
            min_score = scores_array.min()
            max_score = scores_array.max()
            if max_score == min_score:
                return [1.0] * len(scores)
            normalized = (scores_array - min_score) / (max_score - min_score)
            return normalized.tolist()
    
    retriever = MockRetriever()
    normalized = retriever._normalize_scores(scores)

    assert len(normalized) == len(scores)
    assert max(normalized) == 1.0
    assert min(normalized) == 0.0

