"""
Citation formatting utilities.

Converts raw chunk metadata into human-readable citation strings and
formatted source snippets for display in the Streamlit UI.

Also provides APA and IEEE style formatters for the export feature.
"""

from typing import List, Dict, Any


# ─── Main display formatters ───────────────────────────────────────────────────

def format_citations(chunks: List[Dict[str, Any]]) -> str:
    """
    Build a de-duplicated, human-readable citation list from *chunks*.

    Deduplication is (filename, page_number) — if the same page was
    retrieved multiple times, it appears only once.

    Args:
        chunks: Reranked chunk dicts with at minimum ``filename``,
                ``page_number``, and optionally ``rerank_score``.

    Returns:
        Markdown-formatted citation string.
    """
    if not chunks:
        return "*No sources available.*"

    lines: List[str] = ["**Sources used:**\n"]
    seen: set = set()
    citation_num = 1

    for chunk in chunks:
        key = (chunk["filename"], chunk["page_number"])
        if key in seen:
            continue
        seen.add(key)

        rerank_score = chunk.get("rerank_score")
        if rerank_score is not None:
            confidence = _rerank_label(rerank_score)
        else:
            confidence = _l2_label(chunk.get("score", 999.0))

        lines.append(
            f"{citation_num}. **{chunk['filename']}** — "
            f"Page {chunk['page_number']} | Confidence: {confidence}"
        )
        citation_num += 1

    return "\n".join(lines)


def format_source_snippets(
    chunks: List[Dict[str, Any]],
    max_chars: int = 300,
) -> str:
    """
    Format the text preview of each source chunk for the expandable UI panel.

    Args:
        chunks:    Reranked chunks.
        max_chars: Maximum characters to show per snippet.

    Returns:
        Markdown string with one block per chunk.
    """
    if not chunks:
        return ""

    blocks: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        snippet = chunk["text"]
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "…"

        score_part = ""
        if "rerank_score" in chunk:
            score_part = f" | Score: {chunk['rerank_score']:.3f}"

        blocks.append(
            f"**[{i}] {chunk['filename']} — Page {chunk['page_number']}**"
            f"{score_part}\n"
            f"> {snippet}"
        )

    return "\n\n".join(blocks)


# ─── Academic citation formatters ──────────────────────────────────────────────

def build_apa_citation(filename: str, page: int) -> str:
    """
    Return a simple APA-style citation string.

    Real APA requires author, year, title, etc. which are not available
    from the PDF filename alone.  This provides a best-effort format
    suitable for a research chatbot context.

    Example:
        "Attention Is All You Need. (n.d.). Page 5."
    """
    title = _filename_to_title(filename)
    return f"{title}. (n.d.). Page {page}."


def build_ieee_citation(filename: str, page: int, index: int) -> str:
    """
    Return a simple IEEE-style citation string.

    Example:
        "[1] Attention Is All You Need, p. 5."
    """
    title = _filename_to_title(filename)
    return f"[{index}] {title}, p. {page}."


# ─── Internal helpers ──────────────────────────────────────────────────────────

def _filename_to_title(filename: str) -> str:
    """Convert a PDF filename to a readable title string."""
    name = filename
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    name = name.replace("_", " ").replace("-", " ")
    return name.title()


def _rerank_label(score: float) -> str:
    """Convert a Cohere rerank score (0–1, higher = better) to a label."""
    if score >= 0.7:
        return f"High ({score:.2f})"
    if score >= 0.4:
        return f"Medium ({score:.2f})"
    return f"Low ({score:.2f})"


def _l2_label(distance: float) -> str:
    """Convert an L2 distance (lower = better) to a confidence label."""
    if distance < 0.5:
        return "High"
    if distance < 1.5:
        return "Medium"
    return "Low"
