"""
PDF loading and text extraction using PyMuPDF (fitz).

Each page is extracted independently so we can preserve the exact
page number in chunk metadata — critical for accurate citations.
"""

import os
from typing import List, Dict, Any

import fitz  # PyMuPDF

from utils.logger import get_logger
from utils.text_utils import clean_text

logger = get_logger(__name__)


def load_pdf(file_path: str) -> List[Dict[str, Any]]:
    """
    Open *file_path* and return a list of page dicts, one per non-empty page.

    Each page dict contains:
        text         (str)  — cleaned page text
        page_number  (int)  — 1-based page number
        filename     (str)  — basename of the PDF file
        total_pages  (int)  — total number of pages in the document
        file_path    (str)  — absolute path to the file

    Args:
        file_path: Absolute path to the PDF file.

    Returns:
        List of page dicts.  May be empty if no text could be extracted
        (e.g., a fully-scanned / image-only PDF).

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError:        If PyMuPDF cannot open the file (corrupt / wrong type).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"PDF not found: {file_path}")

    filename = os.path.basename(file_path)

    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        raise ValueError(f"Cannot open PDF '{filename}': {exc}") from exc

    total_pages = len(doc)
    logger.info(f"Opened '{filename}' — {total_pages} page(s)")

    pages: List[Dict[str, Any]] = []

    for page_idx in range(total_pages):
        page = doc[page_idx]
        raw_text = page.get_text("text")  # plain text layout
        cleaned = clean_text(raw_text)

        if not cleaned:
            logger.debug(f"  Skipping empty page {page_idx + 1} in '{filename}'")
            continue

        pages.append(
            {
                "text": cleaned,
                "page_number": page_idx + 1,
                "filename": filename,
                "total_pages": total_pages,
                "file_path": file_path,
            }
        )

    doc.close()
    logger.info(
        f"Extracted text from {len(pages)}/{total_pages} page(s) in '{filename}'"
    )
    return pages
