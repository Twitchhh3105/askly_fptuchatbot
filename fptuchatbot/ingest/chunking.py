"""
Intelligent document chunking with support for:
- Semantic chunking (preserves headers and structure)
- Sentence-based chunking (simple overlap)
- Vietnamese text handling
"""

import re
from typing import Any, Dict, List, Optional

import tiktoken

from fptuchatbot.utils.config import get_settings
from fptuchatbot.utils.logging import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """
    Smart document chunking with multiple strategies.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        min_tokens: int = 50,
        use_semantic: bool = True,
        encoding_name: str = "cl100k_base",
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks in characters
            min_tokens: Minimum tokens to keep a chunk
            use_semantic: Use semantic chunking (preserves structure)
            encoding_name: Tiktoken encoding for token counting
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_tokens = min_tokens
        self.use_semantic = use_semantic

        try:
            self.encoding = tiktoken.get_encoding(encoding_name)
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding: {e}")
            self.encoding = None

        self.settings = get_settings()

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.encoding:
            return len(self.encoding.encode(text))
        else:
            # Fallback: approximate as chars/4
            return len(text) // 4

    def detect_headers(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect Vietnamese document headers.

        Supports:
        - Điều X, Khoản X
        - Chương I, Chương 1
        - A., B., C.
        - 1., 2., 3.
        """
        headers = []

        patterns = [
            (r"^(Chương\s+[IVXLCDM\d]+[^\n]*)", 1),  # Chương I, Chương 1
            (r"^(Điều\s+\d+[^\n]*)", 2),  # Điều 1, Điều 2
            (r"^(Khoản\s+\d+[^\n]*)", 3),  # Khoản 1, Khoản 2
            (r"^(Mục\s+\d+[^\n]*)", 3),  # Mục 1, Mục 2
            (r"^([A-Z]\.\s+[^\n]+)", 4),  # A. Header, B. Header
            (r"^(\d+\.\s+[^\n]+)", 5),  # 1. Header, 2. Header
        ]

        lines = text.split("\n")
        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped or len(line_stripped) < 3:
                continue

            for pattern, level in patterns:
                match = re.match(pattern, line_stripped, re.UNICODE)
                if match:
                    headers.append(
                        {
                            "text": match.group(1).strip(),
                            "line_num": line_num,
                            "level": level,
                        }
                    )
                    break

        return headers

    def chunk_semantic(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic chunking that preserves document structure.

        Args:
            text: Text to chunk
            metadata: Optional metadata to attach

        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []

        headers = self.detect_headers(text)
        paragraphs = self._split_paragraphs(text)

        chunks = []
        current_chunk = []
        current_tokens = 0
        current_headers = []  # Stack of active headers

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # Check if paragraph is a header
            is_header = any(h["text"] in para for h in headers)

            if is_header:
                # Update header stack
                for h in headers:
                    if h["text"] in para:
                        # Remove headers of same or lower level
                        current_headers = [ch for ch in current_headers if ch["level"] < h["level"]]
                        current_headers.append(h)
                        break
                continue  # Don't add header to chunk yet

            # Check if adding paragraph exceeds chunk size
            header_text = "\n".join([h["text"] for h in current_headers])
            header_tokens = self.count_tokens(header_text)
            total_tokens = current_tokens + para_tokens + header_tokens

            max_tokens = self.chunk_size // 4  # Convert chars to approximate tokens

            if total_tokens > max_tokens and current_chunk:
                # Save current chunk
                chunk_text = self._build_chunk_text(current_headers, current_chunk)
                chunk_tokens = self.count_tokens(chunk_text)

                if chunk_tokens >= self.min_tokens:
                    chunks.append(
                        {
                            "text": chunk_text,
                            "tokens": chunk_tokens,
                            "headers": [h["text"] for h in current_headers],
                            "metadata": metadata or {},
                        }
                    )

                # Start new chunk with overlap
                overlap_size = len(current_chunk) // 4  # Keep last 25%
                current_chunk = current_chunk[-overlap_size:] + [para]
                current_tokens = sum(self.count_tokens(p) for p in current_chunk)
            else:
                current_chunk.append(para)
                current_tokens += para_tokens

        # Add last chunk
        if current_chunk:
            chunk_text = self._build_chunk_text(current_headers, current_chunk)
            chunk_tokens = self.count_tokens(chunk_text)

            if chunk_tokens >= self.min_tokens:
                chunks.append(
                    {
                        "text": chunk_text,
                        "tokens": chunk_tokens,
                        "headers": [h["text"] for h in current_headers],
                        "metadata": metadata or {},
                    }
                )

        return chunks

    def _build_chunk_text(
        self, headers: List[Dict[str, Any]], paragraphs: List[str]
    ) -> str:
        """Build chunk text with headers."""
        header_text = "\n".join([h["text"] for h in headers])
        para_text = "\n\n".join(paragraphs)

        if header_text:
            return f"{header_text}\n\n{para_text}"
        return para_text

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        # Split by double newlines or more
        paragraphs = re.split(r"\n\s*\n+", text)
        # Clean and filter
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return paragraphs

    def chunk_simple(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Simple overlapping chunks without semantic awareness.

        Args:
            text: Text to chunk
            metadata: Optional metadata

        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # Try to end at sentence boundary
            if end < text_len:
                # Look for sentence ending
                last_period = chunk_text.rfind(".")
                last_question = chunk_text.rfind("?")
                last_exclaim = chunk_text.rfind("!")

                boundary = max(last_period, last_question, last_exclaim)
                if boundary > len(chunk_text) // 2:  # At least halfway through
                    end = start + boundary + 1
                    chunk_text = text[start:end]

            chunk_text = chunk_text.strip()
            chunk_tokens = self.count_tokens(chunk_text)

            if chunk_tokens >= self.min_tokens:
                chunks.append(
                    {
                        "text": chunk_text,
                        "tokens": chunk_tokens,
                        "headers": [],
                        "metadata": metadata or {},
                    }
                )

            # Move forward with overlap
            start = end - self.chunk_overlap

        return chunks

    def chunk_document(
        self, pages_and_texts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Chunk entire document (multiple pages).

        Args:
            pages_and_texts: List of page dictionaries with 'text' field

        Returns:
            List of chunks with metadata
        """
        all_chunks = []
        chunk_id = 1

        for page_info in pages_and_texts:
            page_num = page_info.get("page_number", 0)
            page_text = page_info.get("text", "")

            if not page_text.strip():
                continue

            # Prepare metadata
            metadata = page_info.get("metadata", {})
            metadata["page_number"] = page_num
            metadata["source_file"] = page_info.get("source_file", "unknown")

            # Chunk this page
            if self.use_semantic:
                page_chunks = self.chunk_semantic(page_text, metadata)
            else:
                page_chunks = self.chunk_simple(page_text, metadata)

            # Add chunk IDs and format
            for chunk in page_chunks:
                chunk["chunk_id"] = chunk_id
                chunk["page_number"] = page_num
                chunk["sentence_chunk"] = chunk["text"]  # Alias for compatibility
                chunk["chunk_char_count"] = len(chunk["text"])
                chunk["chunk_word_count"] = len(chunk["text"].split())
                chunk["chunk_token_count"] = chunk["tokens"]

                # Flatten metadata for easier access
                for key, value in metadata.items():
                    if key not in chunk:
                        chunk[key] = value

                all_chunks.append(chunk)
                chunk_id += 1

        logger.info(f"Created {len(all_chunks)} chunks from {len(pages_and_texts)} pages")
        return all_chunks

