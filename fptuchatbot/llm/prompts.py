"""
Prompt templates for RAG system.
"""

from typing import Any, Dict, List


class PromptBuilder:
    """
    Build prompts for different tasks.
    """

    @staticmethod
    def build_answer_prompt(
        query: str,
        context_chunks: List[Dict[str, Any]],
        max_chunks: int = 8,
    ) -> str:
        """
        Build prompt for answer generation.

        Args:
            query: User query
            context_chunks: Retrieved context chunks
            max_chunks: Maximum chunks to include

        Returns:
            Formatted prompt
        """
        # Build context
        context_parts = []
        for i, chunk in enumerate(context_chunks[:max_chunks], 1):
            text = chunk.get("text", chunk.get("sentence_chunk", ""))
            source = chunk.get("source_file", "Unknown")
            page = chunk.get("page_number", "?")

            context_parts.append(
                f"[Trích dẫn {i}] (Nguồn: {source}, Trang {page})\n{text}"
            )

        context_text = "\n\n".join(context_parts)

        prompt = f"""Bạn là trợ lý AI của Trường Đại học FPT, chuyên trả lời các câu hỏi về quy định, chính sách, và thông tin của trường.

Thông tin tham khảo:
{context_text}

Câu hỏi: {query}

Hướng dẫn trả lời:
- Trả lời bằng tiếng Việt, chính xác và chuyên nghiệp
- Chỉ sử dụng thông tin từ tài liệu được cung cấp ở trên
- Nếu không tìm thấy thông tin, hãy nói rõ "Xin lỗi, tôi không tìm thấy thông tin về vấn đề này trong tài liệu hiện có"
- Trả lời ngắn gọn, súc tích nhưng đầy đủ
- Nếu có nhiều điều khoản liên quan, liệt kê rõ ràng
Câu trả lời:"""

        return prompt

    @staticmethod
    def build_query_rewrite_prompt(query: str) -> str:
        """
        Build prompt for query rewriting/expansion.

        Args:
            query: Original query

        Returns:
            Prompt for query rewriting
        """
        prompt = f"""Hãy viết lại câu hỏi sau để tìm kiếm hiệu quả hơn trong cơ sở dữ liệu tài liệu.
Mở rộng các từ viết tắt, thêm từ khóa liên quan, giữ nguyên nghĩa.

Câu hỏi gốc: {query}

Câu hỏi sau khi viết lại (chỉ trả về câu hỏi, không giải thích):"""

        return prompt

    @staticmethod
    def build_system_instruction() -> str:
        """
        Build system instruction for the assistant.

        Returns:
            System instruction string
        """
        return """Bạn là trợ lý AI của Trường Đại học FPT (FPT University).
Nhiệm vụ của bạn là trả lời các câu hỏi của sinh viên, phụ huynh về:
- Quy định, quy chế đào tạo
- Tuyển sinh và nhập học
- Học phí và chính sách hỗ trợ
- Nội quy, kỷ luật
- Ký túc xá và dịch vụ
- Nghiên cứu khoa học
- OJT và đồ án tốt nghiệp

Luôn trả lời bằng tiếng Việt, chính xác, chuyên nghiệp, và thân thiện."""

    @staticmethod
    def build_followup_prompt(
        query: str,
        previous_answer: str,
        context_chunks: List[Dict[str, Any]],
    ) -> str:
        """
        Build prompt for follow-up questions.

        Args:
            query: Follow-up query
            previous_answer: Previous answer
            context_chunks: New context chunks

        Returns:
            Follow-up prompt
        """
        context_text = "\n\n".join([
            f"[{i}] {chunk.get('text', chunk.get('sentence_chunk', ''))}"
            for i, chunk in enumerate(context_chunks[:5], 1)
        ])

        prompt = f"""Câu trả lời trước đó:
{previous_answer}

Thông tin bổ sung:
{context_text}

Câu hỏi tiếp theo: {query}

Hãy trả lời câu hỏi tiếp theo, tham khảo cả câu trả lời trước và thông tin bổ sung.

Câu trả lời:"""

        return prompt

