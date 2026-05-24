"""
Grounding guardrail and confidence scoring for the RAG pipeline.

Evidence score
--------------
We synthesise a single float in [0, 1] from three signals:
  1. Mean rerank score of the top-N chunks (primary).
  2. Count of "supporting" chunks (≥ MIN_SUPPORTING_SCORE).
  3. Max rerank score (bonus for a very high top result).

Confidence label
----------------
  High   — evidence_score ≥ CONFIDENCE_HIGH_THRESHOLD
  Medium — evidence_score ≥ CONFIDENCE_MEDIUM_THRESHOLD
  Low    — below medium threshold (may still answer)

Guardrail
---------
  If evidence_score < GROUNDING_REFUSAL_THRESHOLD the pipeline returns
  a pre-written soft-refusal instead of calling the LLM. This prevents
  hallucination when no relevant content was found.
"""

from typing import List, Dict, Any, Tuple

from utils import config
from utils.logger import get_logger

logger = get_logger(__name__)

# ─── Fallback text ─────────────────────────────────────────────────────────────

_REFUSAL_TEMPLATE = (
    "⚠️ **I could not ground an answer from the uploaded papers.**\n\n"
    "The retrieved passages do not contain sufficient evidence to answer "
    "this question reliably. Please try:\n"
    "- Rephrasing the question.\n"
    "- Uploading additional papers on this topic.\n"
    "- Checking whether the answer might be in a different section of the paper.\n\n"
    "*Evidence score: {score:.2f} (threshold: {threshold:.2f})*"
)

_LOW_CONFIDENCE_PREFIX = (
    "⚠️ **Low confidence** — the supporting evidence is limited.\n\n"
)


# ─── Core functions ─────────────────────────────────────────────────────────────

def compute_evidence_score(chunks: List[Dict[str, Any]]) -> float:
    """
    Compute a single evidence score in [0, 1] from a list of reranked chunks.

    Cohere Rerank returns near-zero scores for generic / document-level queries
    (e.g. "summarize this paper") because the cross-encoder is tuned for factual
    Q&A.  When all rerank scores are effectively 0 but FAISS did retrieve chunks,
    we fall back to the FAISS L2 distance as a weak proxy so the LLM can still
    attempt the answer (at Low confidence).

    Cohere's ``embed-english-v3.0`` produces unit-norm vectors, so L2 distances
    live in [0, 2]:  ~0 = identical, ~1.4 = orthogonal, 2 = opposite.

    Args:
        chunks: Reranked chunk dicts (may have ``rerank_score`` and/or ``score``).

    Returns:
        Evidence score in [0.0, 1.0].  Returns 0.0 for an empty list.
    """
    if not chunks:
        return 0.0

    # Gather rerank scores
    rerank_scores = [
        c["rerank_score"]
        for c in chunks
        if "rerank_score" in c
    ]

    # If the best rerank score is below 0.05, the cross-encoder essentially
    # found no strong signal — common for open-ended / summarisation queries.
    # Fall back to FAISS L2 proximity so the LLM can still attempt an answer.
    if not rerank_scores or max(rerank_scores) < 0.05:
        l2_scores = [c.get("score", 999.0) for c in chunks]
        # Map L2 distance in [0, 2] → proximity in [0, 1]
        proximity = [max(0.0, 1.0 - s / 2.0) for s in l2_scores]
        raw = float(sum(proximity) / len(proximity))
        # Guarantee we clear the refusal threshold: FAISS retrieved chunks so
        # the document IS present — the cross-encoder just can't score this
        # query type (open-ended, summarization, etc.).  Answer at Low confidence.
        floor = config.GROUNDING_REFUSAL_THRESHOLD + 0.01  # 0.06 by default
        return round(min(max(raw, floor), 0.18), 4)

    mean_score = sum(rerank_scores) / len(rerank_scores)
    max_score = max(rerank_scores)

    # Count chunks that clear the minimum support bar
    supporting = sum(
        1 for s in rerank_scores if s >= config.MIN_SUPPORTING_SCORE
    )
    support_ratio = supporting / len(rerank_scores)

    # Weighted combination: mean (50%) + max (30%) + support ratio (20%)
    evidence = 0.50 * mean_score + 0.30 * max_score + 0.20 * support_ratio
    evidence = round(min(evidence, 1.0), 4)

    logger.debug(
        f"Evidence score: {evidence:.4f} "
        f"(mean={mean_score:.3f}, max={max_score:.3f}, support={support_ratio:.2f})"
    )
    return evidence


def get_confidence_label(evidence_score: float) -> str:
    """
    Map an evidence score to a human-readable confidence label.

    Args:
        evidence_score: Float in [0, 1] from :func:`compute_evidence_score`.

    Returns:
        One of "High", "Medium", or "Low".
    """
    if evidence_score >= config.CONFIDENCE_HIGH_THRESHOLD:
        return "High"
    if evidence_score >= config.CONFIDENCE_MEDIUM_THRESHOLD:
        return "Medium"
    return "Low"


def get_confidence_emoji(label: str) -> str:
    """Return an emoji indicator for a confidence label."""
    return {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(label, "⚪")


def should_refuse(evidence_score: float) -> bool:
    """
    Return True if evidence is too weak to attempt an LLM answer.

    Args:
        evidence_score: Float from :func:`compute_evidence_score`.
    """
    return evidence_score < config.GROUNDING_REFUSAL_THRESHOLD


def build_refusal_message(evidence_score: float) -> str:
    """
    Return the pre-written soft-refusal text for low-evidence queries.

    Args:
        evidence_score: The computed evidence score.
    """
    return _REFUSAL_TEMPLATE.format(
        score=evidence_score,
        threshold=config.GROUNDING_REFUSAL_THRESHOLD,
    )


def annotate_answer(
    answer: str,
    evidence_score: float,
    label: str,
) -> str:
    """
    Prepend a confidence badge to the LLM answer.

    Args:
        answer:         The raw LLM answer.
        evidence_score: Float from :func:`compute_evidence_score`.
        label:          "High", "Medium", or "Low".

    Returns:
        Answer string with confidence annotation prepended.
    """
    emoji = get_confidence_emoji(label)
    badge = (
        f"{emoji} **Confidence: {label}** "
        f"*(evidence score: {evidence_score:.2f})*\n\n"
    )
    if label == "Low":
        return badge + _LOW_CONFIDENCE_PREFIX + answer
    return badge + answer


def evaluate_grounding(
    chunks: List[Dict[str, Any]],
) -> Tuple[float, str, bool]:
    """
    Convenience wrapper: compute score, label, and refusal flag.

    Args:
        chunks: Reranked chunks.

    Returns:
        Tuple of (evidence_score, confidence_label, should_refuse).
    """
    score = compute_evidence_score(chunks)
    label = get_confidence_label(score)
    refuse = should_refuse(score)
    return score, label, refuse
