"""
FAISS index management for dense retrieval.
Supports FlatIP and HNSW index types.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from fptuchatbot.utils.config import get_settings
from fptuchatbot.utils.logging import get_logger
from fptuchatbot.utils.timing import Timer

logger = get_logger(__name__)


class FAISSIndex:
    """
    FAISS index wrapper for vector similarity search.
    """

    def __init__(
        self,
        dimension: int,
        index_type: str = "FlatIP",
        metric: str = "inner_product",
        hnsw_m: int = 32,
        hnsw_ef_construction: int = 200,
        hnsw_ef_search: int = 64,
    ):
        """
        Initialize FAISS index.

        Args:
            dimension: Embedding dimension
            index_type: 'FlatIP' or 'HNSW'
            metric: Distance metric ('inner_product' or 'l2')
            hnsw_m: HNSW M parameter (links per node)
            hnsw_ef_construction: HNSW ef_construction
            hnsw_ef_search: HNSW ef_search
        """
        self.dimension = dimension
        self.index_type = index_type
        self.metric = metric
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        self.index = self._create_index()
        self.chunks: List[Dict[str, Any]] = []  # Store chunk metadata

        logger.info(f"Initialized FAISS {index_type} index (dim={dimension}, metric={metric})")

    def _create_index(self) -> faiss.Index:
        """Create FAISS index based on type."""
        if self.index_type == "FlatIP":
            # Flat index with inner product (best for accuracy)
            index = faiss.IndexFlatIP(self.dimension)

        elif self.index_type == "HNSW":
            # HNSW index (fast approximate search)
            index = faiss.IndexHNSWFlat(self.dimension, self.hnsw_m)
            index.hnsw.efConstruction = self.hnsw_ef_construction
            index.hnsw.efSearch = self.hnsw_ef_search

        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")

        return index

    def add(self, embeddings: np.ndarray, chunks: List[Dict[str, Any]]) -> None:
        """
        Add embeddings and chunks to index.

        Args:
            embeddings: Numpy array (N, D)
            chunks: List of chunk dictionaries (length N)
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Embeddings ({len(embeddings)}) and chunks ({len(chunks)}) must have same length"
            )

        # Ensure embeddings are float32
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Normalize for inner product if needed
        if self.metric == "inner_product":
            faiss.normalize_L2(embeddings)

        logger.info(f"Adding {len(embeddings)} vectors to FAISS index")
        with Timer("Add vectors to index"):
            self.index.add(embeddings)

        # Store chunks
        self.chunks.extend(chunks)

        logger.info(f"Total vectors in index: {self.index.ntotal}")

    def search(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for similar vectors.

        Args:
            query_embedding: Query vector (D,) or (1, D)
            top_k: Number of results to return

        Returns:
            Tuple of (scores, indices)
        """
        # Ensure 2D array
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Ensure float32
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)

        # Normalize for inner product
        if self.metric == "inner_product":
            faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, top_k)

        return scores[0], indices[0]

    def search_with_chunks(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Search and return chunks with scores.

        Args:
            query_embedding: Query vector
            top_k: Number of results

        Returns:
            List of dictionaries with chunk and score
        """
        scores, indices = self.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores, indices):
            if idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result["faiss_score"] = float(score)
                result["faiss_rank"] = len(results)
                results.append(result)

        return results

    def save(self, index_path: Path, chunks_path: Optional[Path] = None) -> None:
        """
        Save index and chunks to disk.

        Args:
            index_path: Path to save FAISS index (.faiss)
            chunks_path: Path to save chunks (.pkl). If None, uses index_path with .pkl
        """
        # Save FAISS index
        logger.info(f"Saving FAISS index to {index_path}")
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))

        # Save chunks
        if chunks_path is None:
            chunks_path = index_path.with_suffix(".pkl")

        logger.info(f"Saving {len(self.chunks)} chunks to {chunks_path}")
        with open(chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

        logger.info("FAISS index and chunks saved successfully")

    def load(self, index_path: Path, chunks_path: Optional[Path] = None) -> None:
        """
        Load index and chunks from disk.

        Args:
            index_path: Path to FAISS index (.faiss)
            chunks_path: Path to chunks (.pkl). If None, uses index_path with .pkl
        """
        # Load FAISS index
        logger.info(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(str(index_path))

        # Load chunks
        if chunks_path is None:
            chunks_path = index_path.with_suffix(".pkl")

        logger.info(f"Loading chunks from {chunks_path}")
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)

        logger.info(
            f"Loaded FAISS index with {self.index.ntotal} vectors and {len(self.chunks)} chunks"
        )

    @classmethod
    def from_saved(
        cls, index_path: Path, chunks_path: Optional[Path] = None
    ) -> "FAISSIndex":
        """
        Create FAISSIndex from saved files.

        Args:
            index_path: Path to FAISS index
            chunks_path: Path to chunks

        Returns:
            Loaded FAISSIndex instance
        """
        # Load index to get dimension
        index = faiss.read_index(str(index_path))
        dimension = index.d

        # Create instance
        instance = cls(dimension=dimension)
        instance.load(index_path, chunks_path)

        return instance

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "index_type": self.index_type,
            "dimension": self.dimension,
            "num_vectors": self.index.ntotal,
            "num_chunks": len(self.chunks),
            "metric": self.metric,
        }

