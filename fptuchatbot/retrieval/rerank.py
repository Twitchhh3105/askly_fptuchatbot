"""
Cross-encoder reranking for improving retrieval quality.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import CrossEncoder

from fptuchatbot.utils.config import get_settings
from fptuchatbot.utils.logging import get_logger
from fptuchatbot.utils.timing import Timer

logger = get_logger(__name__)


class CrossEncoderReranker:
    """
    Cross-encoder reranker for improving retrieval quality.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        """
        Initialize reranker.

        Args:
            model_name: Cross-encoder model name
            device: Device ('cpu' or 'cuda')
            batch_size: Batch size for prediction
        """
        settings = get_settings()

        self.model_name = model_name or settings.reranker_model
        self.device = device or settings.reranker_device
        self.batch_size = batch_size

        logger.info(f"Loading reranker model: {self.model_name} on {self.device}")

        try:
            with Timer(f"Load reranker {self.model_name}"):
                self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info("Reranker loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            raise

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        text_field: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder.

        Args:
            query: Query string
            results: List of result dictionaries
            top_k: Number of top results to return (None = all)
            text_field: Field containing text to rerank

        Returns:
            Reranked results with scores
        """
        if not results:
            return []

        logger.info(f"Reranking {len(results)} results")

        with Timer("Reranking"):
            # Prepare query-document pairs
            pairs = []
            for result in results:
                doc_text = result.get(text_field, result.get("sentence_chunk", ""))
                pairs.append([query, doc_text])

            # Get reranking scores
            scores = self.model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )

            # Add scores to results
            for result, score in zip(results, scores):
                result["rerank_score"] = float(score)

            # Sort by rerank score
            reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

            # Return top-k
            if top_k:
                reranked = reranked[:top_k]

        logger.info(f"Reranking complete, returning {len(reranked)} results")
        return reranked

    def score_pairs(
        self, query_doc_pairs: List[Tuple[str, str]]
    ) -> np.ndarray:
        """
        Score query-document pairs.

        Args:
            query_doc_pairs: List of (query, document) tuples

        Returns:
            Array of scores
        """
        scores = self.model.predict(
            query_doc_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return scores


from typing import Optional

