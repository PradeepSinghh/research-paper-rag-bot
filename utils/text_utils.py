"""
Text cleaning and normalisation utilities.

Pure-Python helpers that make raw PDF text cleaner before chunking.
No external dependencies — keeps tests fast and deterministic.
"""

import re

from utils.logger import get_logger

logger = get_logger(__name__)


def clean_text(text: str) -> str:
    """
    Clean raw text extracted from a PDF page.

    Steps applied (in order):
    1. Remove ASCII control characters (except \\n and \\t).
    2. Normalise line endings to \\n.
    3. Collapse runs of 3+ newlines into two (paragraph break).
    4. Collapse runs of spaces / tabs into a single space.
    5. Strip leading/trailing whitespace from every line.
    6. Strip leading/trailing whitespace from the whole result.

    Args:
        text: Raw text string from PyMuPDF.

    Returns:
        Cleaned text string.  Empty string if input is empty / None.
    """
    if not text:
        return ""

    # 1. Remove control characters (keep \n = 0x0A, \t = 0x09)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 2. Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Max two consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # 5. Strip each line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def truncate_text(text: str, max_chars: int = 300) -> str:
    """
    Return at most *max_chars* characters, appending "…" if the text was cut.

    Args:
        text:      Input string.
        max_chars: Maximum number of characters to keep.

    Returns:
        Possibly truncated string.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def word_count(text: str) -> int:
    """Return an approximate word count by splitting on whitespace."""
    return len(text.split())


def remove_references_section(text: str) -> str:
    """
    Attempt to strip the References / Bibliography section from the *end*
    of a paper.  Only removes text that appears in the last 40 % of the
    document (to avoid removing inline reference lists).

    Args:
        text: Full paper text.

    Returns:
        Text with trailing References section removed (if found).
    """
    patterns = [
        r"\n\s*references\s*\n",
        r"\n\s*bibliography\s*\n",
        r"\n\s*works cited\s*\n",
    ]

    threshold = int(len(text) * 0.6)

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.start() > threshold:
            logger.debug(f"Trimming references section at char {match.start()}")
            text = text[: match.start()]
            break

    return text.strip()
