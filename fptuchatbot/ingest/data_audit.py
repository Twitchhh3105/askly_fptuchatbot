"""
Data audit module for analyzing PDF corpus quality and coverage.
Generates manifest, quality metrics, and coverage reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from fptuchatbot.ingest.pdf_loader import PDFLoader
from fptuchatbot.utils.config import get_settings
from fptuchatbot.utils.logging import get_logger
from fptuchatbot.utils.timing import Timer

logger = get_logger(__name__)


class DataAuditor:
    """
    Audit data quality and coverage for RAG system.
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data auditor.

        Args:
            data_dir: Directory containing PDF files
        """
        self.settings = get_settings()
        self.data_dir = data_dir or self.settings.get_data_dir()
        self.pdf_loader = PDFLoader()

    def create_manifest(self) -> List[Dict[str, Any]]:
        """
        Create manifest of all PDF files in data directory.

        Returns:
            List of file metadata dictionaries
        """
        logger.info(f"Creating manifest for {self.data_dir}")

        pdf_files = sorted(self.data_dir.glob("*.pdf"))
        manifest = []

        for pdf_path in pdf_files:
            try:
                # Get file stats
                stat = pdf_path.stat()

                # Get PDF info
                pdf_info = self.pdf_loader.get_pdf_info(pdf_path)

                # Estimate topic from filename
                topic = self._estimate_topic(pdf_path.name)

                # Estimate language (simplified)
                language = "vi"  # Vietnamese default

                entry = {
                    "file_path": str(pdf_path),
                    "filename": pdf_path.name,
                    "file_type": "pdf",
                    "size_bytes": stat.st_size,
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "num_pages": pdf_info.get("num_pages", 0),
                    "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "language": language,
                    "topic": topic,
                    "metadata": pdf_info.get("metadata", {}),
                }

                manifest.append(entry)

            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")

        logger.info(f"Created manifest with {len(manifest)} files")
        return manifest

    def _estimate_topic(self, filename: str) -> str:
        """Estimate document topic from filename."""
        filename_lower = filename.lower()

        topic_keywords = {
            "tuyển sinh": ["tuyen sinh", "nhap hoc", "xet tuyen"],
            "học phí": ["hoc phi", "hoc phi", "thu phi", "chi phi"],
            "quy chế đào tạo": ["quy che dao tao", "dao tao", "chuong trinh"],
            "nội quy": ["noi quy", "quy dinh", "quy tac"],
            "khen thưởng": ["khen thuong", "hoc bong", "thuong"],
            "nghiên cứu khoa học": ["nghien cuu", "nckh", "khoa hoc"],
            "ký túc xá": ["ky tuc xa", "ktx", "nha o"],
            "OJT": ["ojt", "thuc tap", "do an"],
            "thông tin chung": ["thong tin", "gioi thieu"],
            "thạc sĩ": ["thac si", "cao hoc"],
            "liên hệ": ["lien he", "contact"],
        }

        for topic, keywords in topic_keywords.items():
            if any(kw in filename_lower for kw in keywords):
                return topic

        return "khác"

    def analyze_quality(self, manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze data quality metrics.

        Args:
            manifest: File manifest

        Returns:
            Quality metrics dictionary
        """
        logger.info("Analyzing data quality...")

        total_files = len(manifest)
        parse_success = 0
        parse_failed = []
        ocr_needed = 0
        tables_extracted = 0
        total_pages = 0

        for entry in manifest:
            pdf_path = Path(entry["file_path"])
            total_pages += entry.get("num_pages", 0)

            try:
                # Try to load PDF
                pages_data = self.pdf_loader.load(pdf_path)

                if pages_data:
                    parse_success += 1

                    # Check if OCR was used
                    ocr_used = any(
                        p.get("extraction_method", "").endswith("ocr") for p in pages_data
                    )
                    if ocr_used:
                        ocr_needed += 1

                else:
                    parse_failed.append(entry["filename"])

            except Exception as e:
                logger.error(f"Failed to analyze {entry['filename']}: {e}")
                parse_failed.append(entry["filename"])

        quality_metrics = {
            "total_files": total_files,
            "total_pages": total_pages,
            "parse_success_count": parse_success,
            "parse_success_rate": round(parse_success / total_files * 100, 2) if total_files > 0 else 0,
            "parse_failed_count": len(parse_failed),
            "parse_failed_files": parse_failed,
            "ocr_needed_count": ocr_needed,
            "ocr_rate": round(ocr_needed / total_files * 100, 2) if total_files > 0 else 0,
            "tables_extracted_count": tables_extracted,
            "avg_pages_per_file": round(total_pages / total_files, 1) if total_files > 0 else 0,
        }

        logger.info(f"Quality analysis: {parse_success}/{total_files} files parsed successfully")
        return quality_metrics

    def probe_coverage(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        bm25_index=None,
        probe_questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Probe coverage with representative questions.

        Args:
            chunks: Indexed chunks
            bm25_index: BM25 index for search
            probe_questions: List of test questions

        Returns:
            Coverage metrics
        """
        logger.info("Probing coverage with representative questions...")

        if probe_questions is None:
            probe_questions = self._get_default_probe_questions()

        if bm25_index is None or chunks is None:
            logger.warning("No index provided, skipping coverage probe")
            return {
                "total_questions": len(probe_questions),
                "probe_questions": probe_questions,
                "coverage_note": "Index not built yet - run after ingestion",
            }

        coverage_results = []

        for question in probe_questions:
            try:
                # Search with BM25
                results = bm25_index.search(question, top_k=10)

                # Check if relevant results found
                has_results = len(results) > 0
                top_score = results[0]["bm25_score"] if results else 0

                coverage_results.append(
                    {
                        "question": question,
                        "has_results": has_results,
                        "num_results": len(results),
                        "top_score": float(top_score),
                        "top_result": results[0].get("text", "")[:200] if results else None,
                    }
                )

            except Exception as e:
                logger.error(f"Coverage probe failed for '{question}': {e}")

        # Calculate metrics
        answered = sum(1 for r in coverage_results if r["has_results"])
        coverage_rate = round(answered / len(probe_questions) * 100, 2)

        coverage_metrics = {
            "total_questions": len(probe_questions),
            "answered_count": answered,
            "coverage_rate": coverage_rate,
            "unanswered_questions": [
                r["question"] for r in coverage_results if not r["has_results"]
            ],
            "detailed_results": coverage_results,
        }

        logger.info(f"Coverage: {answered}/{len(probe_questions)} questions ({coverage_rate}%)")
        return coverage_metrics

    def _get_default_probe_questions(self) -> List[str]:
        """Get default probe questions for FPT University."""
        return [
            # Tuyển sinh
            "Điều kiện xét tuyển vào trường là gì?",
            "Học phí ngành công nghệ thông tin bao nhiêu?",
            "Các ngành đào tạo của trường?",
            "Thời gian nộp hồ sơ xét tuyển?",
            # Đào tạo
            "Quy chế đào tạo của trường?",
            "Điều kiện tốt nghiệp?",
            "Thang điểm đánh giá như thế nào?",
            "Quy định về học lại môn học?",
            # OJT và thực tập
            "OJT là gì?",
            "Thời gian thực hiện OJT?",
            "Đồ án tốt nghiệp có yêu cầu gì?",
            # Nội quy
            "Quy định về vi phạm kỷ luật?",
            "Quy định về trang phục?",
            "Quy tắc ứng xử trong trường?",
            # Học phí và hỗ trợ
            "Chính sách miễn giảm học phí?",
            "Học bổng của trường?",
            "Quy định về nộp học phí?",
            # Khen thưởng
            "Điều kiện nhận học bổng?",
            "Quy định khen thưởng cuối kỳ?",
            # Ký túc xá
            "Nội quy ký túc xá?",
            "Đăng ký ký túc xá như thế nào?",
            "Giá phòng ký túc xá?",
            # Nghiên cứu
            "Quy định về nghiên cứu khoa học?",
            "Hỗ trợ tham dự hội nghị khoa học?",
            "Quản lý đề tài nghiên cứu?",
            # Thạc sĩ
            "Quy chế đào tạo thạc sĩ?",
            "Điều kiện nhập học thạc sĩ?",
            # Thông tin chung
            "Thông tin liên hệ của trường?",
            "Cơ sở vật chất của trường?",
            "Công nghệ và hệ thống của trường?",
            # Chi tiết hơn
            "Quy định về chuyển ngành?",
            "Quy định về bảo lưu kết quả học tập?",
            "Quy định về khiếu nại điểm thi?",
            "Các hình thức kỷ luật sinh viên?",
            "Quy định về đạo đức nghiên cứu?",
            "Tiêu chuẩn năng lực nghiên cứu?",
            "Quyền sở hữu trí tuệ trong nghiên cứu?",
            "Nội quy phòng thi?",
            "Quy định về làm bài thi?",
        ]

    def generate_report(
        self,
        manifest: List[Dict[str, Any]],
        quality_metrics: Dict[str, Any],
        coverage_metrics: Optional[Dict[str, Any]] = None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive data report.

        Args:
            manifest: File manifest
            quality_metrics: Quality analysis results
            coverage_metrics: Coverage probe results
            output_path: Path to save report JSON

        Returns:
            Complete report dictionary
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "data_directory": str(self.data_dir),
            "manifest": manifest,
            "quality_metrics": quality_metrics,
            "coverage_metrics": coverage_metrics or {},
            "summary": {
                "total_files": len(manifest),
                "total_size_mb": sum(f["size_mb"] for f in manifest),
                "total_pages": sum(f["num_pages"] for f in manifest),
                "parse_success_rate": quality_metrics.get("parse_success_rate", 0),
                "coverage_rate": coverage_metrics.get("coverage_rate", 0) if coverage_metrics else 0,
            },
        }

        # Save to file if path provided
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            logger.info(f"Data report saved to {output_path}")

        return report

    def run_full_audit(
        self,
        chunks: Optional[List[Dict[str, Any]]] = None,
        bm25_index=None,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run complete audit process.

        Args:
            chunks: Optional chunks for coverage probe
            bm25_index: Optional BM25 index for coverage probe
            output_path: Optional path to save report

        Returns:
            Complete audit report
        """
        logger.info("Starting full data audit...")

        with Timer("Full data audit"):
            # Create manifest
            manifest = self.create_manifest()

            # Analyze quality
            quality_metrics = self.analyze_quality(manifest)

            # Probe coverage
            coverage_metrics = self.probe_coverage(chunks, bm25_index)

            # Generate report
            report = self.generate_report(
                manifest, quality_metrics, coverage_metrics, output_path
            )

        logger.info("Data audit completed")
        return report

