"""
Document chunking with sliding-window overlap.

Pages are split into overlapping character-level chunks.  Breaking at
paragraph and sentence boundaries keeps chunks semantically coherent and
avoids cutting sentences mid-way — which confuses both the embedder and
the LLM.

Each chunk is a dict that carries all the source metadata needed for
citation generation.
"""

from typing import List, Dict, Any

from utils.logger import get_logger
from utils import config

logger = get_logger(__name__)


def chunk_pages(
    pages: List[Dict[str, Any]],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
    min_length: int = config.MIN_CHUNK_LENGTH,
) -> List[Dict[str, Any]]:
    """
    Split a list of page dicts into overlapping text chunks.

    Each returned chunk dict contains:
        chunk_id    (int)  — monotonically increasing ID across all chunks
        text        (str)  — chunk text
        filename    (str)  — source PDF filename
        page_number (int)  — 1-based page number the chunk starts on
        total_pages (int)  — total pages in the source document
        file_path   (str)  — absolute path to the source PDF

    Args:
        pages:         Output of loader.load_pdf().
        chunk_size:    Target character length per chunk.
        chunk_overlap: Overlap in characters between consecutive chunks.
        min_length:    Discard chunks shorter than this (e.g., stray page numbers).

    Returns:
        Flat list of chunk dicts.
    """
    chunks: List[Dict[str, Any]] = []
    chunk_id = 0

    for page in pages:
        page_text = page["text"]
        raw_chunks = _split_text(page_text, chunk_size, chunk_overlap)

        for chunk_text in raw_chunks:
            stripped = chunk_text.strip()
            if len(stripped) < min_length:
                continue

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": stripped,
                    "filename": page["filename"],
                    "page_number": page["page_number"],
                    "total_pages": page["total_pages"],
                    "file_path": page.get("file_path", ""),
                }
            )
            chunk_id += 1

    logger.info(
        f"Created {len(chunks)} chunk(s) from {len(pages)} page(s) "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split *text* into overlapping substrings of approximately *chunk_size*
    characters, with *overlap* characters shared between consecutive chunks.

    Prefers to break at paragraph (\\n\\n) > sentence (". ") > newline boundaries
    so chunks are semantically whole rather than mid-sentence.

    Args:
        text:       Input string.
        chunk_size: Target size in characters.
        overlap:    Overlap in characters.

    Returns:
        List of chunk strings (may be empty).
    """
    if not text.strip():
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        if end >= len(text):
            # Last chunk — take the rest
            chunk = text[start:]
        else:
            # Snap to a natural boundary within a search window
            end = _find_break(text, end, search_window=100)
            chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        # Slide forward, preserving overlap
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1  # guarantee progress
        start = next_start

    return chunks


def _find_break(text: str, pos: int, search_window: int = 100) -> int:
    """
    Look *search_window* characters backward from *pos* for a good break point.

    Priority: paragraph break > sentence end > newline > original position.

    Args:
        text:          Full text string.
        pos:           Desired break position.
        search_window: Characters to search backward.

    Returns:
        Adjusted break position.
    """
    window_start = max(0, pos - search_window)
    segment = text[window_start:pos]

    # 1. Paragraph break
    idx = segment.rfind("\n\n")
    if idx != -1:
        return window_start + idx + 2

    # 2. Sentence break
    idx = segment.rfind(". ")
    if idx != -1:
        return window_start + idx + 2

    # 3. Any newline
    idx = segment.rfind("\n")
    if idx != -1:
        return window_start + idx + 1

    return pos
