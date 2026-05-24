"""
RAG evaluation metrics for the trust dashboard.

All metrics are computed locally — no external API needed.

Metrics
-------
hit_rate_at_k
    Fraction of retrieved chunks that contain at least one keyword from
    the query.  Proxy for retrieval relevance.

context_precision
    Fraction of retrieved chunks whose rerank score ≥ threshold.
    Proxy for how well the reranker filtered noise.

mean_rerank_score
    Average rerank score across the top-N chunks.

rerank_improvement
    Difference between mean score of the top-N reranked chunks and the
    mean score of the raw (pre-rerank) L2 chunks.  Positive means
    reranking helped.

fallback_rate
    Across all tracked queries, fraction that triggered the soft refusal.

answer_length
    Number of words in the generated answer (readability proxy).

groundedness_ratio
    Fraction of supported chunks (rerank_score ≥ MIN_SUPPORTING_SCORE)
    out of total chunks — measures how well the LLM context was grounded.
"""

from __future__ import annotations

import re
from typing import List, Dict, Any

from utils import config
from utils.logger import get_logger

logger = get_logger(__name__)


# ─── Per-query metrics ─────────────────────────────────────────────────────────

def hit_rate_at_k(
    query: str,
    chunks: List[Dict[str, Any]],
    k: int | None = None,
) -> float:
    """
    Compute hit rate: fraction of chunks (up to k) that contain at least
    one non-stopword keyword from *query*.

    Args:
        query:  User's question.
        chunks: Reranked chunks (the top-N that were sent to the LLM).
        k:      Cap on chunks to consider.  Defaults to len(chunks).

    Returns:
        Float in [0, 1].  0.0 if no chunks provided.
    """
    if not chunks:
        return 0.0

    _STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on",
        "of", "to", "and", "or", "what", "how", "why", "when",
        "which", "do", "does", "did", "for", "with", "that",
    }

    keywords = [
        w.lower()
        for w in re.findall(r"\w+", query)
        if w.lower() not in _STOPWORDS and len(w) > 2
    ]

    if not keywords:
        return 1.0  # Cannot evaluate without keywords — assume hit

    pool = chunks[:k] if k else chunks
    hits = sum(
        1
        for c in pool
        if any(kw in c.get("text", "").lower() for kw in keywords)
    )
    return round(hits / len(pool), 4)


def context_precision(
    chunks: List[Dict[str, Any]],
    threshold: float | None = None,
) -> float:
    """
    Fraction of chunks whose rerank_score ≥ *threshold*.

    Args:
        chunks:    Reranked chunk dicts.
        threshold: Score threshold.  Defaults to config.MIN_SUPPORTING_SCORE.

    Returns:
        Float in [0, 1].  0.0 if no chunks provided.
    """
    if not chunks:
        return 0.0
    thr = threshold if threshold is not None else config.MIN_SUPPORTING_SCORE
    good = sum(1 for c in chunks if c.get("rerank_score", 0.0) >= thr)
    return round(good / len(chunks), 4)


def mean_rerank_score(chunks: List[Dict[str, Any]]) -> float:
    """Average rerank_score across *chunks*.  Returns 0.0 if absent."""
    scores = [c.get("rerank_score", 0.0) for c in chunks]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def rerank_improvement(
    raw_chunks: List[Dict[str, Any]],
    reranked_chunks: List[Dict[str, Any]],
) -> float:
    """
    Estimate how much reranking improved chunk relevance.

    Compares mean rerank score of reranked chunks to mean rerank score
    of the *same* chunks before they were sorted by the reranker.

    In practice we compare the mean rerank score of the kept chunks vs
    the mean rerank score of the chunks that were dropped (lower-ranked).

    Args:
        raw_chunks:     Full list from FAISS (pre-rerank), may be empty.
        reranked_chunks: Top-N returned by the reranker.

    Returns:
        Signed delta float.  Positive = reranking raised the average score.
    """
    if not raw_chunks or not reranked_chunks:
        return 0.0

    kept_mean = mean_rerank_score(reranked_chunks)

    # raw_chunks may not have rerank scores; if so, delta is undefined
    raw_with_scores = [c for c in raw_chunks if "rerank_score" in c]
    if not raw_with_scores:
        return 0.0

    raw_mean = mean_rerank_score(raw_with_scores)
    return round(kept_mean - raw_mean, 4)


def answer_length(answer: str) -> int:
    """Word count of the generated answer."""
    return len(answer.split())


def groundedness_ratio(chunks: List[Dict[str, Any]]) -> float:
    """
    Fraction of chunks that clear the minimum supporting score bar.

    Args:
        chunks: Top-N chunks used for generation.

    Returns:
        Float in [0, 1].
    """
    if not chunks:
        return 0.0
    supporting = sum(
        1
        for c in chunks
        if c.get("rerank_score", 0.0) >= config.MIN_SUPPORTING_SCORE
    )
    return round(supporting / len(chunks), 4)


# ─── Session-level aggregation ────────────────────────────────────────────────

def compute_query_metrics(
    query: str,
    raw_chunks: List[Dict[str, Any]],
    reranked_chunks: List[Dict[str, Any]],
    answer: str,
    was_refused: bool,
) -> Dict[str, Any]:
    """
    Return a dict of all per-query evaluation metrics.

    Args:
        query:            The user's question.
        raw_chunks:       Chunks from FAISS (pre-rerank).
        reranked_chunks:  Top-N chunks from the reranker.
        answer:           Generated LLM answer text.
        was_refused:      Whether the grounding guardrail triggered.

    Returns:
        Dict with metric keys and values.
    """
    return {
        "query": query[:120],
        "hit_rate_at_k": hit_rate_at_k(query, reranked_chunks),
        "context_precision": context_precision(reranked_chunks),
        "mean_rerank_score": mean_rerank_score(reranked_chunks),
        "rerank_improvement": rerank_improvement(raw_chunks, reranked_chunks),
        "groundedness_ratio": groundedness_ratio(reranked_chunks),
        "answer_length_words": answer_length(answer),
        "chunks_retrieved": len(raw_chunks),
        "chunks_after_rerank": len(reranked_chunks),
        "was_refused": was_refused,
    }


def aggregate_session_metrics(
    history: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Aggregate per-query metrics across the session.

    Args:
        history: List of dicts returned by :func:`compute_query_metrics`.

    Returns:
        Dict with session-level averages and counts.
    """
    if not history:
        return {}

    n = len(history)
    avg = lambda key: round(sum(h[key] for h in history) / n, 4)

    return {
        "total_queries": n,
        "fallback_rate": round(sum(h["was_refused"] for h in history) / n, 4),
        "avg_hit_rate": avg("hit_rate_at_k"),
        "avg_context_precision": avg("context_precision"),
        "avg_mean_rerank_score": avg("mean_rerank_score"),
        "avg_rerank_improvement": avg("rerank_improvement"),
        "avg_groundedness_ratio": avg("groundedness_ratio"),
        "avg_answer_length": avg("answer_length_words"),
    }
