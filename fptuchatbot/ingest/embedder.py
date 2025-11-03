"""
Embedding generation using SentenceTransformers.
Supports bge-m3 (primary) and multilingual-e5-base (fallback).
"""

import hashlib
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sentence_transformers import SentenceTransformer

from fptuchatbot.utils.config import get_settings
from fptuchatbot.utils.logging import get_logger
from fptuchatbot.utils.timing import Timer

logger = get_logger(__name__)


class Embedder:
    """
    Embedding generator with caching support.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialize embedder.

        Args:
            model_name: Model name (e.g., 'BAAI/bge-m3')
            device: Device ('cpu' or 'cuda')
            batch_size: Batch size for encoding
            cache_dir: Directory for caching embeddings
        """
        settings = get_settings()

        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self.batch_size = batch_size
        self.cache_dir = cache_dir or settings.get_cache_dir()

        logger.info(f"Loading embedding model: {self.model_name} on {self.device}")

        try:
            with Timer(f"Load embedding model {self.model_name}"):
                self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded model with dimension: {self.embedding_dim}")
        except Exception as e:
            logger.error(f"Failed to load {self.model_name}: {e}")
            # Try fallback model
            logger.info("Trying fallback model: intfloat/multilingual-e5-base")
            self.model_name = "intfloat/multilingual-e5-base"
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode texts to embeddings.

        Args:
            texts: Single text or list of texts
            normalize: Normalize embeddings to unit length
            show_progress: Show progress bar

        Returns:
            Numpy array of embeddings (N, D)
        """
        if isinstance(texts, str):
            texts = [texts]

        with Timer(f"Encode {len(texts)} texts"):
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

        return embeddings

    def encode_chunks(
        self, chunks: List[Dict[str, Any]], text_field: str = "text"
    ) -> np.ndarray:
        """
        Encode chunks (list of dictionaries).

        Args:
            chunks: List of chunk dictionaries
            text_field: Field name containing text to embed

        Returns:
            Numpy array of embeddings (N, D)
        """
        texts = [chunk.get(text_field, chunk.get("sentence_chunk", "")) for chunk in chunks]

        # Filter empty texts
        valid_texts = [t for t in texts if t.strip()]
        if len(valid_texts) != len(texts):
            logger.warning(
                f"Filtered {len(texts) - len(valid_texts)} empty texts from {len(texts)} chunks"
            )

        logger.info(f"Encoding {len(valid_texts)} chunks")
        embeddings = self.encode(valid_texts, show_progress=True)

        return embeddings

    def encode_with_cache(
        self, texts: Union[str, List[str]], cache_key: Optional[str] = None
    ) -> np.ndarray:
        """
        Encode with file-based caching.

        Args:
            texts: Texts to encode
            cache_key: Cache key (if None, computes hash from texts)

        Returns:
            Embeddings array
        """
        if cache_key is None:
            # Compute cache key from texts
            if isinstance(texts, str):
                cache_key = hashlib.md5(texts.encode()).hexdigest()
            else:
                cache_key = hashlib.md5("".join(texts).encode()).hexdigest()

        cache_file = self.cache_dir / f"embeddings_{cache_key}.pkl"

        # Check cache
        if cache_file.exists():
            logger.info(f"Loading embeddings from cache: {cache_key[:8]}...")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        # Compute embeddings
        embeddings = self.encode(texts)

        # Save to cache
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump(embeddings, f)
        logger.info(f"Cached embeddings: {cache_key[:8]}...")

        return embeddings

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            emb1: First embedding
            emb2: Second embedding

        Returns:
            Similarity score
        """
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def batch_similarity(self, query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
        """
        Compute similarity between query and multiple documents.

        Args:
            query_emb: Query embedding (D,)
            doc_embs: Document embeddings (N, D)

        Returns:
            Similarity scores (N,)
        """
        # Inner product (assuming normalized embeddings)
        scores = np.dot(doc_embs, query_emb)
        return scores

