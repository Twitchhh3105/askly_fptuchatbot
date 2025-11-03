"""
Table extraction from PDFs using Camelot.
Supports both lattice (with borders) and stream (without borders) modes.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fptuchatbot.utils.logging import get_logger
from fptuchatbot.utils.timing import Timer

logger = get_logger(__name__)


class TableExtractor:
    """Extract tables from PDF files using Camelot."""

    def __init__(
        self,
        flavor: str = "lattice",
        strip_text: str = "\n",
    ):
        """
        Initialize table extractor.

        Args:
            flavor: 'lattice' (tables with borders) or 'stream' (without borders)
            strip_text: Characters to strip from cells
        """
        self.flavor = flavor
        self.strip_text = strip_text

    def extract_tables(
        self, pdf_path: Path, pages: Optional[str] = "all"
    ) -> List[Dict[str, Any]]:
        """
        Extract tables from PDF.

        Args:
            pdf_path: Path to PDF file
            pages: Page numbers to extract ('all', '1', '1,2,3', '1-5')

        Returns:
            List of table dictionaries with page, data, and markdown
        """
        try:
            import camelot

            logger.info(f"Extracting tables from {pdf_path.name} (flavor={self.flavor})")

            with Timer(f"Table extraction - {pdf_path.name}"):
                tables = camelot.read_pdf(
                    str(pdf_path),
                    flavor=self.flavor,
                    pages=pages,
                    strip_text=self.strip_text,
                )

            extracted_tables = []
            for i, table in enumerate(tables):
                # Convert to markdown
                markdown = self._table_to_markdown(table.df)

                table_data = {
                    "table_index": i,
                    "page": table.page,
                    "rows": table.shape[0],
                    "cols": table.shape[1],
                    "accuracy": table.accuracy,
                    "data": table.df.to_dict("records"),
                    "markdown": markdown,
                }
                extracted_tables.append(table_data)

            logger.info(f"Extracted {len(extracted_tables)} tables from {pdf_path.name}")
            return extracted_tables

        except ImportError:
            logger.warning("Camelot not installed. Skipping table extraction.")
            return []
        except Exception as e:
            logger.warning(f"Table extraction failed for {pdf_path.name}: {e}")
            # Try alternative flavor
            if self.flavor == "lattice":
                logger.info("Retrying with 'stream' flavor...")
                self.flavor = "stream"
                return self.extract_tables(pdf_path, pages)
            return []

    def _table_to_markdown(self, df) -> str:
        """
        Convert pandas DataFrame to markdown table.

        Args:
            df: Pandas DataFrame

        Returns:
            Markdown string
        """
        try:
            import pandas as pd

            # Clean column names
            df.columns = [str(col).strip() for col in df.columns]

            # Convert to markdown
            markdown = df.to_markdown(index=False)
            return markdown

        except Exception as e:
            logger.warning(f"Failed to convert table to markdown: {e}")
            return str(df)

    def extract_tables_with_fallback(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract tables with automatic fallback between lattice and stream.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of extracted tables
        """
        # Try lattice first (more accurate for bordered tables)
        self.flavor = "lattice"
        tables = self.extract_tables(pdf_path)

        # If no tables found, try stream
        if not tables:
            logger.info(f"No tables found with lattice, trying stream for {pdf_path.name}")
            self.flavor = "stream"
            tables = self.extract_tables(pdf_path)

        return tables

    def merge_tables_with_text(
        self, pages_data: List[Dict[str, Any]], tables: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge extracted tables back into page data.

        Args:
            pages_data: List of page dictionaries from PDF loader
            tables: List of extracted tables

        Returns:
            Updated pages_data with tables merged
        """
        # Group tables by page
        tables_by_page = {}
        for table in tables:
            page_num = table["page"] - 1  # 0-indexed
            if page_num not in tables_by_page:
                tables_by_page[page_num] = []
            tables_by_page[page_num].append(table)

        # Add tables to corresponding pages
        for page_data in pages_data:
            page_num = page_data["page_number"]
            if page_num in tables_by_page:
                page_tables = tables_by_page[page_num]
                page_data["tables"] = page_tables
                page_data["num_tables"] = len(page_tables)

                # Append table markdown to text
                table_markdowns = [
                    f"\n\n### Table {t['table_index'] + 1}\n{t['markdown']}"
                    for t in page_tables
                ]
                page_data["text"] += "".join(table_markdowns)
            else:
                page_data["tables"] = []
                page_data["num_tables"] = 0

        return pages_data

