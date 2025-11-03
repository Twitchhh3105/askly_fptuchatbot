"""
BM25 sparse retrieval for Vietnamese text.
Uses rank_bm25 with Vietnamese-aware tokenization.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from fptuchatbot.utils.config import get_settings
from fptuchatbot.utils.logging import get_logger
from fptuchatbot.utils.timing import Timer

logger = get_logger(__name__)


class BM25Index:
    """
    BM25 index for sparse retrieval with Vietnamese tokenization.
    """

    def __init__(
        self,
        tokenizer: Optional[str] = "simple",
        use_stopwords: bool = False,
    ):
        """
        Initialize BM25 index.

        Args:
            tokenizer: Tokenization method ('simple', 'underthesea')
            use_stopwords: Whether to use Vietnamese stopwords
        """
        self.tokenizer_type = tokenizer
        self.use_stopwords = use_stopwords
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

        # Vietnamese stopwords
        self.stopwords = set()
        if use_stopwords:
            self.stopwords = self._load_stopwords()

        logger.info(f"Initialized BM25 index with {tokenizer} tokenizer")

    def _load_stopwords(self) -> set:
        """Load Vietnamese stopwords."""
        # Common Vietnamese stopwords
        stopwords = {
            "và", "của", "có", "được", "là", "trong", "cho", "với", "này",
            "đó", "những", "các", "để", "hay", "hoặc", "không", "cũng",
            "được", "làm", "từ", "theo", "như", "đến", "về", "trên", "khi",
            "nếu", "bởi", "tại", "đã", "sẽ", "nên", "thì", "mà", "nhưng",
        }
        return stopwords

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        text = text.lower().strip()

        if self.tokenizer_type == "underthesea":
            try:
                from underthesea import word_tokenize
                tokens = word_tokenize(text, format="text").split()
            except Exception as e:
                logger.warning(f"Underthesea tokenization failed: {e}, using simple tokenizer")
                tokens = text.split()
        else:
            # Simple whitespace tokenizer
            tokens = text.split()

        # Remove stopwords
        if self.use_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]

        # Remove very short tokens
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    def build(self, chunks: List[Dict[str, Any]], text_field: str = "text") -> None:
        """
        Build BM25 index from chunks.

        Args:
            chunks: List of chunk dictionaries
            text_field: Field name containing text
        """
        logger.info(f"Building BM25 index from {len(chunks)} chunks")

        self.chunks = chunks

        # Tokenize corpus
        with Timer("Tokenize corpus"):
            self.tokenized_corpus = []
            for chunk in chunks:
                text = chunk.get(text_field, chunk.get("sentence_chunk", ""))
                tokens = self.tokenize(text)
                self.tokenized_corpus.append(tokens)

        # Build BM25
        with Timer("Build BM25 index"):
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info("BM25 index built successfully")

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search using BM25.

        Args:
            query: Query string
            top_k: Number of results to return

        Returns:
            List of chunks with BM25 scores
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built yet. Call build() first.")

        # Tokenize query
        query_tokens = self.tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        # Build results
        results = []
        for rank, idx in enumerate(top_indices):
            if idx < len(self.chunks) and scores[idx] > 0:
                result = self.chunks[idx].copy()
                result["bm25_score"] = float(scores[idx])
                result["bm25_rank"] = rank
                results.append(result)

        return results

    def save(self, index_path: Path) -> None:
        """
        Save BM25 index to disk.

        Args:
            index_path: Path to save index (.pkl)
        """
        logger.info(f"Saving BM25 index to {index_path}")

        index_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "bm25": self.bm25,
            "chunks": self.chunks,
            "tokenized_corpus": self.tokenized_corpus,
            "tokenizer_type": self.tokenizer_type,
            "use_stopwords": self.use_stopwords,
        }

        with open(index_path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"BM25 index with {len(self.chunks)} chunks saved")

    def load(self, index_path: Path) -> None:
        """
        Load BM25 index from disk.

        Args:
            index_path: Path to saved index (.pkl)
        """
        logger.info(f"Loading BM25 index from {index_path}")

        with open(index_path, "rb") as f:
            data = pickle.load(f)

        self.bm25 = data["bm25"]
        self.chunks = data["chunks"]
        self.tokenized_corpus = data["tokenized_corpus"]
        self.tokenizer_type = data.get("tokenizer_type", "simple")
        self.use_stopwords = data.get("use_stopwords", False)

        if self.use_stopwords:
            self.stopwords = self._load_stopwords()

        logger.info(f"Loaded BM25 index with {len(self.chunks)} chunks")

    @classmethod
    def from_saved(cls, index_path: Path) -> "BM25Index":
        """
        Load BM25Index from saved file.

        Args:
            index_path: Path to saved index

        Returns:
            Loaded BM25Index instance
        """
        instance = cls()
        instance.load(index_path)
        return instance

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            "num_chunks": len(self.chunks),
            "tokenizer": self.tokenizer_type,
            "use_stopwords": self.use_stopwords,
            "avg_tokens": np.mean([len(tokens) for tokens in self.tokenized_corpus])
            if self.tokenized_corpus
            else 0,
        }

