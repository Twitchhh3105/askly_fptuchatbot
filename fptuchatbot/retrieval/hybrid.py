"""
Hybrid retrieval combining BM25 (sparse) and FAISS (dense) with score fusion.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from fptuchatbot.ingest.bm25 import BM25Index
from fptuchatbot.ingest.embedder import Embedder
from fptuchatbot.ingest.index_faiss import FAISSIndex
from fptuchatbot.utils.config import get_settings
from fptuchatbot.utils.logging import get_logger
from fptuchatbot.utils.timing import Timer

logger = get_logger(__name__)


class HybridRetriever:
    """
    Hybrid retrieval combining sparse (BM25) and dense (FAISS) search.
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        faiss_index: FAISSIndex,
        embedder: Embedder,
        top_k_bm25: int = 50,
        top_k_faiss: int = 50,
        bm25_weight: float = 0.5,
        faiss_weight: float = 0.5,
    ):
        """
        Initialize hybrid retriever.

        Args:
            bm25_index: BM25 index for sparse retrieval
            faiss_index: FAISS index for dense retrieval
            embedder: Embedder for query encoding
            top_k_bm25: Number of results from BM25
            top_k_faiss: Number of results from FAISS
            bm25_weight: Weight for BM25 scores in fusion
            faiss_weight: Weight for FAISS scores in fusion
        """
        self.bm25_index = bm25_index
        self.faiss_index = faiss_index
        self.embedder = embedder
        self.top_k_bm25 = top_k_bm25
        self.top_k_faiss = top_k_faiss
        self.bm25_weight = bm25_weight
        self.faiss_weight = faiss_weight
        self.settings = get_settings()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        return_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid search.

        Args:
            query: Query string
            top_k: Number of final results (if None, uses settings)
            return_scores: Include individual scores in results

        Returns:
            List of retrieved chunks with scores
        """
        if top_k is None:
            top_k = self.top_k_bm25 + self.top_k_faiss

        logger.info(f"Hybrid retrieval for query: '{query[:50]}...'")

        with Timer("Hybrid retrieval"):
            # BM25 retrieval
            with Timer("BM25 search"):
                bm25_results = self.bm25_index.search(query, self.top_k_bm25)
            logger.info(f"BM25 retrieved {len(bm25_results)} results")

            # FAISS retrieval
            with Timer("FAISS search"):
                query_embedding = self.embedder.encode(query)
                faiss_results = self.faiss_index.search_with_chunks(
                    query_embedding, self.top_k_faiss
                )
            logger.info(f"FAISS retrieved {len(faiss_results)} results")

            # Merge and deduplicate
            with Timer("Merge and deduplicate"):
                merged_results = self._merge_results(
                    bm25_results, faiss_results, return_scores
                )

            # Sort by fused score and return top-k
            merged_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            final_results = merged_results[:top_k]

        logger.info(f"Hybrid retrieval returned {len(final_results)} results")
        return final_results

    def _merge_results(
        self,
        bm25_results: List[Dict[str, Any]],
        faiss_results: List[Dict[str, Any]],
        return_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Merge and deduplicate BM25 and FAISS results.

        Args:
            bm25_results: Results from BM25
            faiss_results: Results from FAISS
            return_scores: Include individual scores

        Returns:
            Merged and deduplicated results
        """
        # Normalize scores to [0, 1]
        bm25_scores_norm = self._normalize_scores(
            [r["bm25_score"] for r in bm25_results]
        )
        faiss_scores_norm = self._normalize_scores(
            [r["faiss_score"] for r in faiss_results]
        )

        # Create lookup by chunk_id
        merged = {}

        # Add BM25 results
        for result, norm_score in zip(bm25_results, bm25_scores_norm):
            chunk_id = self._get_chunk_id(result)
            merged[chunk_id] = {
                **result,
                "bm25_score_norm": norm_score,
                "faiss_score_norm": 0.0,
                "sources": ["bm25"],
            }

        # Add/merge FAISS results
        for result, norm_score in zip(faiss_results, faiss_scores_norm):
            chunk_id = self._get_chunk_id(result)

            if chunk_id in merged:
                # Update FAISS score
                merged[chunk_id]["faiss_score"] = result.get("faiss_score", 0)
                merged[chunk_id]["faiss_score_norm"] = norm_score
                merged[chunk_id]["sources"].append("faiss")
            else:
                # New entry
                merged[chunk_id] = {
                    **result,
                    "bm25_score_norm": 0.0,
                    "faiss_score_norm": norm_score,
                    "sources": ["faiss"],
                }

        # Compute hybrid scores
        for chunk_id, result in merged.items():
            bm25_contrib = self.bm25_weight * result["bm25_score_norm"]
            faiss_contrib = self.faiss_weight * result["faiss_score_norm"]
            result["hybrid_score"] = bm25_contrib + faiss_contrib

            # Clean up if not returning scores
            if not return_scores:
                result.pop("bm25_score_norm", None)
                result.pop("faiss_score_norm", None)

        return list(merged.values())

    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """
        Normalize scores to [0, 1] using min-max normalization.

        Args:
            scores: Raw scores

        Returns:
            Normalized scores
        """
        if not scores:
            return []

        scores_array = np.array(scores)
        min_score = scores_array.min()
        max_score = scores_array.max()

        if max_score == min_score:
            return [1.0] * len(scores)

        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized.tolist()

    def _get_chunk_id(self, result: Dict[str, Any]) -> str:
        """
        Get unique identifier for chunk.

        Args:
            result: Result dictionary

        Returns:
            Unique chunk identifier
        """
        # Try multiple fields to create unique ID
        chunk_id = result.get("chunk_id")
        if chunk_id:
            return str(chunk_id)

        # Fallback: use combination of source and page
        source = result.get("source_file", "unknown")
        page = result.get("page_number", 0)
        text_hash = hash(result.get("text", result.get("sentence_chunk", ""))[:100])

        return f"{source}::{page}::{text_hash}"

