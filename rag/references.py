"""
Reference / bibliography extraction from PDF pages.

Heuristics-based approach — no ML model needed:
1. Detect a "References" or "Bibliography" heading page.
2. Collect all text after that heading.
3. Split into individual reference entries using numbered patterns
   ([1], 1., etc.) or hanging-indent line grouping.

The output is a list of dicts suitable for direct display in the UI
and inclusion in the JSON export.
"""

import re
from typing import List, Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Heading detection ─────────────────────────────────────────────────────────

_REF_HEADING_RE = re.compile(
    r"^\s*(references|bibliography|works cited|literature cited|citations)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Common reference entry starters:
#   [1] Smith ...
#   1. Smith ...
#   1) Smith ...
# Multiline pattern — matches start of each line.
# Handles: [1] text,  1. text,  1) text
_ENTRY_START_RE = re.compile(r"(?m)^\s*(?:\[\d{1,3}\]\s+|\d{1,3}[.)]\s+)")


def _find_ref_section_start(pages: List[Dict[str, Any]]) -> Optional[int]:
    """
    Return the page index (0-based) where the reference section begins,
    or None if not found.
    """
    # Search from the back — references are almost always at the end
    for idx in range(len(pages) - 1, -1, -1):
        text = pages[idx].get("text", "")
        if _REF_HEADING_RE.search(text):
            logger.debug(
                f"Reference section heading found on page "
                f"{pages[idx]['page_number']}"
            )
            return idx
    return None


def _split_into_entries(text: str) -> List[str]:
    """
    Split a block of reference text into individual reference entries.

    Tries numbered entries first; falls back to double-newline splitting.
    """
    # Use finditer to locate each numbered entry start (multiline)
    matches = list(_ENTRY_START_RE.finditer(text))
    if len(matches) >= 2:
        entries: List[str] = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            entry = text[start:end].strip()
            if entry:
                entries.append(entry)
        if entries:
            return entries

    # Fallback: split on blank lines
    raw = [e.strip() for e in re.split(r"\n{2,}", text)]
    return [e for e in raw if len(e) > 20]  # discard very short fragments


def extract_references(
    pages: List[Dict[str, Any]],
    filename: str,
) -> List[Dict[str, Any]]:
    """
    Extract bibliography entries from *pages*.

    Args:
        pages:    Page dicts from :func:`rag.loader.load_pdf`.
        filename: PDF filename (used to tag each entry).

    Returns:
        List of reference entry dicts, each with keys:
          - ``index``    (int)   — 1-based position in the list.
          - ``text``     (str)   — full entry text.
          - ``filename`` (str)   — source PDF.
          - ``page``     (int)   — page number where the section starts.
    """
    if not pages:
        return []

    start_idx = _find_ref_section_start(pages)
    if start_idx is None:
        logger.info(f"No reference section found in '{filename}'")
        return []

    ref_page_number = pages[start_idx]["page_number"]

    # Collect all text from the heading page onwards
    raw_text_parts: List[str] = []
    for page in pages[start_idx:]:
        raw_text_parts.append(page.get("text", ""))
    combined = "\n".join(raw_text_parts)

    # Trim everything before the heading
    match = _REF_HEADING_RE.search(combined)
    if match:
        combined = combined[match.end():]

    entries = _split_into_entries(combined)
    logger.info(f"Extracted {len(entries)} reference(s) from '{filename}'")

    return [
        {
            "index": i,
            "text": entry,
            "filename": filename,
            "page": ref_page_number,
        }
        for i, entry in enumerate(entries, 1)
    ]


def format_references_markdown(references: List[Dict[str, Any]]) -> str:
    """
    Format a list of reference dicts as a Markdown numbered list.

    Args:
        references: Output of :func:`extract_references`.

    Returns:
        Markdown string.  Empty string if no references.
    """
    if not references:
        return ""
    lines = [f"{r['index']}. {r['text']}" for r in references]
    return "\n".join(lines)
