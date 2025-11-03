"""
PDF Loader with support for Vietnamese text, tables, and OCR.
Uses Unstructured (hi_res + OCR) as primary, PyMuPDF as fallback.
"""

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import fitz  # PyMuPDF
from PIL import Image

from fptuchatbot.utils.config import get_settings
from fptuchatbot.utils.logging import get_logger
from fptuchatbot.utils.timing import Timer

logger = get_logger(__name__)


class PDFLoader:
    """
    PDF loader with multi-strategy extraction:
    1. Unstructured (hi_res + OCR for Vietnamese)
    2. PyMuPDF fallback
    3. OCR for scanned pages
    """

    def __init__(
        self,
        strategy: str = "hi_res",
        ocr_languages: str = "vie+eng",
        remove_footer: bool = True,
        footer_threshold: int = 70,
    ):
        """
        Initialize PDF loader.

        Args:
            strategy: Unstructured strategy ('hi_res', 'fast', 'ocr_only')
            ocr_languages: Tesseract language codes (e.g., 'vie+eng')
            remove_footer: Whether to remove footer regions
            footer_threshold: Pixels from bottom to consider as footer
        """
        self.strategy = strategy
        self.ocr_languages = ocr_languages
        self.remove_footer = remove_footer
        self.footer_threshold = footer_threshold
        self.settings = get_settings()

    def load_pdf_unstructured(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Load PDF using Unstructured library (best for complex PDFs).

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of page dictionaries with text and metadata
        """
        try:
            from unstructured.partition.pdf import partition_pdf

            logger.info(f"Loading PDF with Unstructured: {pdf_path.name}")

            with Timer(f"Unstructured extraction - {pdf_path.name}"):
                elements = partition_pdf(
                    filename=str(pdf_path),
                    strategy=self.strategy,
                    languages=[self.ocr_languages.split("+")[0]],  # Primary language
                    infer_table_structure=True,
                    include_page_breaks=True,
                )

            # Group elements by page
            pages_data = []
            current_page = 0
            current_text = []

            for element in elements:
                # Check if new page
                if hasattr(element, "metadata") and element.metadata.page_number:
                    page_num = element.metadata.page_number - 1  # 0-indexed

                    if page_num != current_page and current_text:
                        # Save previous page
                        page_text = "\n".join(current_text)
                        pages_data.append(
                            {
                                "page_number": current_page,
                                "text": page_text,
                                "page_char_count": len(page_text),
                                "page_word_count": len(page_text.split()),
                                "extraction_method": "unstructured",
                            }
                        )
                        current_text = []
                        current_page = page_num

                # Add element text
                text = str(element).strip()
                if text:
                    current_text.append(text)

            # Add last page
            if current_text:
                page_text = "\n".join(current_text)
                pages_data.append(
                    {
                        "page_number": current_page,
                        "text": page_text,
                        "page_char_count": len(page_text),
                        "page_word_count": len(page_text.split()),
                        "extraction_method": "unstructured",
                    }
                )

            logger.info(
                f"Extracted {len(pages_data)} pages with Unstructured from {pdf_path.name}"
            )
            return pages_data

        except Exception as e:
            logger.warning(f"Unstructured failed for {pdf_path.name}: {e}")
            return []

    def load_pdf_pymupdf(
        self, pdf_path: Path, use_ocr: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Load PDF using PyMuPDF (fallback method).

        Args:
            pdf_path: Path to PDF file
            use_ocr: Whether to use OCR for scanned pages

        Returns:
            List of page dictionaries with text and metadata
        """
        logger.info(f"Loading PDF with PyMuPDF: {pdf_path.name}")

        try:
            doc = fitz.open(str(pdf_path))
            pages_data = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_height = page.rect.height

                # Check if page needs OCR
                page_text = page.get_text()
                is_scanned = len(page_text.strip()) < 50

                if use_ocr and is_scanned:
                    logger.info(
                        f"Page {page_num + 1} appears scanned, attempting OCR..."
                    )
                    page_text = self._extract_text_with_ocr(page)
                else:
                    # Extract text with optional footer removal
                    if self.remove_footer:
                        lines = []
                        for block in page.get_text("blocks"):
                            x0, y0, x1, y1, text, *_ = block
                            # Skip footer region
                            if y1 > page_height - self.footer_threshold:
                                continue
                            lines.append(text.strip())
                        page_text = "\n".join([line for line in lines if line])
                    else:
                        page_text = page.get_text()

                # Clean text
                page_text = page_text.replace("\n", " ").strip()

                pages_data.append(
                    {
                        "page_number": page_num,
                        "text": page_text,
                        "page_char_count": len(page_text),
                        "page_word_count": len(page_text.split()),
                        "extraction_method": "pymupdf_ocr" if is_scanned else "pymupdf",
                    }
                )

            doc.close()
            logger.info(f"Extracted {len(pages_data)} pages from {pdf_path.name}")
            return pages_data

        except Exception as e:
            logger.error(f"PyMuPDF failed for {pdf_path.name}: {e}")
            return []

    def _extract_text_with_ocr(self, page: fitz.Page) -> str:
        """
        Extract text using OCR (for scanned pages).

        Args:
            page: PyMuPDF page object

        Returns:
            Extracted text
        """
        try:
            import pytesseract

            # Convert page to image at 300 DPI for better OCR (4.17x zoom)
            pix = page.get_pixmap(matrix=fitz.Matrix(4.17, 4.17))
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))

            # Run OCR
            text = pytesseract.image_to_string(img, lang=self.ocr_languages)
            return text.strip()

        except Exception as e:
            logger.warning(f"OCR failed: {e}")
            return ""

    def load(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Load PDF with automatic fallback strategy.

        Tries Unstructured first, falls back to PyMuPDF if needed.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of page dictionaries
        """
        if not pdf_path.exists():
            logger.error(f"PDF not found: {pdf_path}")
            return []

        # Try Unstructured first (best quality)
        pages_data = self.load_pdf_unstructured(pdf_path)

        # Fallback to PyMuPDF if Unstructured fails or returns empty
        if not pages_data:
            logger.info(f"Falling back to PyMuPDF for {pdf_path.name}")
            pages_data = self.load_pdf_pymupdf(pdf_path, use_ocr=True)

        # Add source file metadata
        for page_data in pages_data:
            page_data["source_file"] = pdf_path.name
            page_data["source_path"] = str(pdf_path)

        return pages_data

    def load_multiple(self, pdf_paths: List[Path]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load multiple PDFs.

        Args:
            pdf_paths: List of PDF file paths

        Returns:
            Dictionary mapping filename to list of page data
        """
        results = {}

        for pdf_path in pdf_paths:
            pages_data = self.load(pdf_path)
            if pages_data:
                results[pdf_path.name] = pages_data

        logger.info(
            f"Loaded {len(results)} PDFs with total {sum(len(p) for p in results.values())} pages"
        )
        return results

    def get_pdf_info(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Get PDF metadata information.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary with PDF metadata
        """
        try:
            doc = fitz.open(str(pdf_path))
            metadata = {
                "filename": pdf_path.name,
                "filepath": str(pdf_path),
                "num_pages": len(doc),
                "file_size_bytes": pdf_path.stat().st_size,
                "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
                "metadata": doc.metadata,
            }
            doc.close()
            return metadata

        except Exception as e:
            logger.error(f"Failed to get PDF info for {pdf_path}: {e}")
            return {
                "filename": pdf_path.name,
                "filepath": str(pdf_path),
                "error": str(e),
            }

