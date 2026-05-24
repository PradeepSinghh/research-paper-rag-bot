"""
Paper comparison mode.

Runs independent retrieve → rerank pipelines for two papers then asks
the Groq LLM to produce a structured side-by-side comparison.

Each paper gets its own retrieval pass so the model always has balanced
evidence from both sides — it does not inadvertently favour whichever
paper happened to rank higher in a combined search.
"""

from typing import List, Dict, Any, Tuple

from rag.retriever import retrieve
from rag.reranker import rerank
from rag.generator import get_groq_client, build_context
from rag.vectorstore import FAISSVectorStore
from utils.logger import get_logger
from utils import config

logger = get_logger(__name__)

# ── System prompt for comparison mode ──────────────────────────────────────────

_COMPARE_SYSTEM_PROMPT = (
    "You are a research assistant specialised in comparing academic papers. "
    "You receive excerpts from Paper A and Paper B.\n\n"
    "Output structure (use these exact headings):\n"
    "## Methodology\n"
    "## Results & Findings\n"
    "## Limitations\n"
    "## Key Differences\n\n"
    "Rules:\n"
    "1. Attribute every claim: use [Paper A: filename, Page N] or "
    "[Paper B: filename, Page N].\n"
    "2. Be objective. Do not favour one paper over the other.\n"
    "3. Only use facts present in the provided context.\n"
    "4. If a section cannot be addressed from the context, say so explicitly."
)


def compare_papers(
    query: str,
    paper_a: str,
    paper_b: str,
    store: FAISSVectorStore,
    top_k: int = config.TOP_K_RETRIEVE,
    top_n: int = config.TOP_K_RERANK,
) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compare two papers on a given question or topic.

    Steps for each paper:
    1. retrieve() — filter FAISS results to that paper only.
    2. rerank()   — reorder by cross-encoder relevance.
    3. Combine both contexts and call Groq with the comparison prompt.

    Args:
        query:   Comparison question (e.g., "How do they handle overfitting?").
        paper_a: Filename of the first paper.
        paper_b: Filename of the second paper.
        store:   Populated FAISSVectorStore.
        top_k:   Chunks to retrieve per paper before reranking.
        top_n:   Chunks to keep per paper after reranking.

    Returns:
        Tuple of (answer_text, chunks_a, chunks_b).
        ``chunks_a`` and ``chunks_b`` can be used to render per-paper
        source snippets in the UI.
    """
    logger.info(
        f"compare_papers | '{paper_a}' vs '{paper_b}' | query='{query[:60]}'"
    )

    # ── Retrieve + rerank per paper ────────────────────────────────────────────
    raw_a = retrieve(query, store, top_k=top_k, paper_filter=paper_a)
    raw_b = retrieve(query, store, top_k=top_k, paper_filter=paper_b)

    chunks_a = rerank(query, raw_a, top_n=top_n) if raw_a else []
    chunks_b = rerank(query, raw_b, top_n=top_n) if raw_b else []

    if not chunks_a and not chunks_b:
        return (
            "Could not find relevant content in either paper for this query.",
            [],
            [],
        )

    # ── Build labelled context blocks ──────────────────────────────────────────
    context_a = (
        f"=== Paper A: {paper_a} ===\n\n{build_context(chunks_a)}"
        if chunks_a
        else f"=== Paper A: {paper_a} ===\n\nNo relevant content found."
    )
    context_b = (
        f"=== Paper B: {paper_b} ===\n\n{build_context(chunks_b)}"
        if chunks_b
        else f"=== Paper B: {paper_b} ===\n\nNo relevant content found."
    )

    full_context = f"{context_a}\n\n{'─' * 60}\n\n{context_b}"

    user_message = (
        f"Context:\n\n{full_context}\n\n"
        f"Comparison question: {query}\n\n"
        "Using only the context above, compare the two papers "
        "following the required output structure."
    )

    # ── Call Groq ──────────────────────────────────────────────────────────────
    client = get_groq_client()

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=[
            {"role": "system", "content": _COMPARE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.1,
        max_tokens=2048,
    )

    answer: str = response.choices[0].message.content or ""
    logger.info(f"Comparison answer: {len(answer)} character(s)")

    return answer, chunks_a, chunks_b
