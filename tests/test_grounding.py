"""
Tests for the grounding guardrail and confidence labelling module.
"""

import pytest
from rag.grounding import (
    compute_evidence_score,
    get_confidence_label,
    get_confidence_emoji,
    should_refuse,
    build_refusal_message,
    annotate_answer,
    evaluate_grounding,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _chunk(rerank_score: float, score: float = 1.0):
    return {
        "text": "Some paper text for testing purposes here.",
        "filename": "paper.pdf",
        "page_number": 1,
        "rerank_score": rerank_score,
        "score": score,
    }


# ── compute_evidence_score ─────────────────────────────────────────────────────

class TestComputeEvidenceScore:
    def test_empty_returns_zero(self):
        assert compute_evidence_score([]) == 0.0

    def test_high_rerank_yields_high_score(self):
        chunks = [_chunk(0.9), _chunk(0.85), _chunk(0.8)]
        score = compute_evidence_score(chunks)
        assert score >= 0.6, f"Expected high score, got {score}"

    def test_low_rerank_yields_low_score(self):
        chunks = [_chunk(0.05), _chunk(0.02)]
        score = compute_evidence_score(chunks)
        assert score < 0.35, f"Expected low score, got {score}"

    def test_score_capped_at_one(self):
        chunks = [_chunk(1.0), _chunk(1.0), _chunk(1.0)]
        assert compute_evidence_score(chunks) <= 1.0

    def test_no_rerank_score_uses_l2_proxy(self):
        chunks = [
            {"text": "text", "filename": "f.pdf", "page_number": 1, "score": 0.5},
        ]
        score = compute_evidence_score(chunks)
        assert 0.0 <= score <= 1.0

    def test_zero_rerank_scores_use_l2_fallback(self):
        """Rerank scores below 0.05 (common for summarization queries) should
        fall back to L2 distance and return a score that clears the refusal
        threshold — FAISS retrieved chunks so the document IS present."""
        chunks = [
            {"text": "text", "filename": "f.pdf", "page_number": 1,
             "rerank_score": 0.007, "score": 0.8},
            {"text": "more", "filename": "f.pdf", "page_number": 2,
             "rerank_score": 0.002, "score": 0.9},
        ]
        score = compute_evidence_score(chunks)
        # Must clear the refusal threshold (0.06 floor) and be capped at 0.18
        assert score >= 0.06
        assert score <= 0.18

    def test_single_perfect_chunk(self):
        score = compute_evidence_score([_chunk(1.0)])
        assert score > 0.5

    def test_mixed_scores(self):
        chunks = [_chunk(0.8), _chunk(0.1), _chunk(0.4)]
        score = compute_evidence_score(chunks)
        assert 0.0 < score < 1.0


# ── get_confidence_label ───────────────────────────────────────────────────────

class TestGetConfidenceLabel:
    def test_high(self):
        assert get_confidence_label(0.9) == "High"

    def test_medium(self):
        assert get_confidence_label(0.35) == "Medium"

    def test_low(self):
        assert get_confidence_label(0.03) == "Low"

    def test_exact_high_threshold(self):
        # 0.55 is the default CONFIDENCE_HIGH_THRESHOLD
        assert get_confidence_label(0.55) == "High"

    def test_exact_medium_threshold(self):
        # 0.20 is the default CONFIDENCE_MEDIUM_THRESHOLD
        assert get_confidence_label(0.20) == "Medium"

    def test_just_below_medium(self):
        assert get_confidence_label(0.19) == "Low"


# ── get_confidence_emoji ───────────────────────────────────────────────────────

class TestGetConfidenceEmoji:
    def test_high(self):
        assert get_confidence_emoji("High") == "🟢"

    def test_medium(self):
        assert get_confidence_emoji("Medium") == "🟡"

    def test_low(self):
        assert get_confidence_emoji("Low") == "🔴"

    def test_unknown(self):
        assert get_confidence_emoji("Unknown") == "⚪"


# ── should_refuse ──────────────────────────────────────────────────────────────

class TestShouldRefuse:
    def test_low_evidence_triggers_refusal(self):
        assert should_refuse(0.02) is True

    def test_medium_evidence_no_refusal(self):
        assert should_refuse(0.5) is False

    def test_exactly_at_threshold_no_refusal(self):
        # Default threshold is 0.05
        assert should_refuse(0.05) is False

    def test_just_below_threshold(self):
        assert should_refuse(0.04) is True


# ── build_refusal_message ──────────────────────────────────────────────────────

class TestBuildRefusalMessage:
    def test_contains_score(self):
        msg = build_refusal_message(0.12)
        assert "0.12" in msg

    def test_contains_refusal_indicator(self):
        msg = build_refusal_message(0.05)
        assert "could not ground" in msg.lower() or "⚠️" in msg


# ── annotate_answer ────────────────────────────────────────────────────────────

class TestAnnotateAnswer:
    def test_high_confidence_badge(self):
        result = annotate_answer("My answer.", 0.8, "High")
        assert "High" in result
        assert "🟢" in result

    def test_low_confidence_adds_warning(self):
        result = annotate_answer("My answer.", 0.1, "Low")
        assert "Low" in result
        assert "🔴" in result

    def test_original_answer_preserved(self):
        result = annotate_answer("Original content.", 0.7, "High")
        assert "Original content." in result


# ── evaluate_grounding ─────────────────────────────────────────────────────────

class TestEvaluateGrounding:
    def test_returns_tuple(self):
        chunks = [_chunk(0.8), _chunk(0.7)]
        score, label, refuse = evaluate_grounding(chunks)
        assert isinstance(score, float)
        assert label in ("High", "Medium", "Low")
        assert isinstance(refuse, bool)

    def test_empty_chunks_refuse(self):
        _, _, refuse = evaluate_grounding([])
        assert refuse is True

    def test_high_evidence_no_refuse(self):
        chunks = [_chunk(0.9), _chunk(0.85)]
        _, _, refuse = evaluate_grounding(chunks)
        assert refuse is False
