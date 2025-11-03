"""
Google Gemini API client for answer generation.
"""

import os
from typing import Any, Dict, List, Optional

import google.generativeai as genai

from fptuchatbot.utils.config import get_settings
from fptuchatbot.utils.logging import get_logger
from fptuchatbot.utils.timing import Timer

logger = get_logger(__name__)


class GeminiClient:
    """
    Google Gemini API client wrapper.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key (if None, reads from settings/env)
            model_name: Model name (if None, uses settings default)
        """
        settings = get_settings()

        self.api_key = api_key or settings.gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or settings.gemini_model

        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            logger.warning("Gemini API key not set. Please set GEMINI_API_KEY in .env")
            self.configured = False
            return

        # Configure Gemini
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(self.model_name)
            self.configured = True
            logger.info(f"Gemini client initialized with model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {e}")
            self.configured = False

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate text completion.

        Args:
            prompt: User prompt
            system_instruction: Optional system instruction
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response

        Returns:
            Generated text
        """
        if not self.configured:
            return "Error: Gemini API not configured. Please set GEMINI_API_KEY."

        try:
            generation_config = {
                "temperature": temperature,
            }
            if max_tokens:
                generation_config["max_output_tokens"] = max_tokens

            with Timer(f"Gemini generation"):
                # Create chat session
                chat = self.model.start_chat(history=[])

                # Add system instruction if provided
                if system_instruction:
                    full_prompt = f"{system_instruction}\n\n{prompt}"
                else:
                    full_prompt = prompt

                # Generate response
                response = chat.send_message(
                    full_prompt,
                    generation_config=generation_config,
                )

                return response.text

        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def generate_with_context(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        system_instruction: Optional[str] = None,
        temperature: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Generate answer with retrieved context.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            system_instruction: Optional system instruction
            temperature: Sampling temperature

        Returns:
            Dictionary with answer and metadata
        """
        # Build context from chunks
        context_text = self._build_context(context_chunks)

        # Build prompt
        prompt = f"""Dựa trên thông tin sau đây, hãy trả lời câu hỏi của người dùng.

Thông tin tham khảo:
{context_text}

Câu hỏi: {query}

Hướng dẫn trả lời:
- Trả lời bằng tiếng Việt
- Chỉ sử dụng thông tin từ tài liệu được cung cấp
- Nếu không tìm thấy thông tin, hãy nói rõ "Tôi không tìm thấy thông tin về vấn đề này trong tài liệu"
- Trích dẫn nguồn khi có thể (tên file, trang)
- Trả lời ngắn gọn, rõ ràng, chính xác

Trả lời:"""

        # Generate answer
        answer = self.generate(
            prompt,
            system_instruction=system_instruction,
            temperature=temperature,
        )

        # Build citations
        citations = self._build_citations(context_chunks)

        return {
            "answer": answer,
            "citations": citations,
            "num_sources": len(context_chunks),
            "context": context_text[:500] + "..." if len(context_text) > 500 else context_text,
        }

    def _build_context(self, chunks: List[Dict[str, Any]], max_chunks: int = 8) -> str:
        """
        Build context string from chunks.

        Args:
            chunks: Context chunks
            max_chunks: Maximum number of chunks to include

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, chunk in enumerate(chunks[:max_chunks], 1):
            text = chunk.get("text", chunk.get("sentence_chunk", ""))
            source = chunk.get("source_file", "Unknown")
            page = chunk.get("page_number", "?")

            context_parts.append(
                f"[{i}] (Nguồn: {source}, Trang {page})\n{text}"
            )

        return "\n\n".join(context_parts)

    def _build_citations(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Build citation list from chunks.

        Args:
            chunks: Context chunks

        Returns:
            List of citation dictionaries
        """
        citations = []

        for chunk in chunks:
            source = chunk.get("source_file", "Unknown")
            page = chunk.get("page_number", "?")
            score = chunk.get("rerank_score") or chunk.get("hybrid_score") or chunk.get("faiss_score") or 0

            citations.append({
                "source": source,
                "page": str(page),
                "score": f"{score:.3f}",
            })

        return citations

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
    ) -> str:
        """
        Multi-turn chat.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature

        Returns:
            Assistant response
        """
        if not self.configured:
            return "Error: Gemini API not configured."

        try:
            # Convert messages to Gemini format
            chat = self.model.start_chat(history=[])

            # Send messages
            for msg in messages[:-1]:  # All but last
                if msg["role"] == "user":
                    chat.send_message(msg["content"])

            # Send last message and get response
            response = chat.send_message(
                messages[-1]["content"],
                generation_config={"temperature": temperature},
            )

            return response.text

        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return f"Error: {str(e)}"

