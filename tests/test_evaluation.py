"""
Tests for the RAG evaluation metrics module.
"""

import pytest
from rag.evaluation import (
    hit_rate_at_k,
    context_precision,
    mean_rerank_score,
    rerank_improvement,
    answer_length,
    groundedness_ratio,
    compute_query_metrics,
    aggregate_session_metrics,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _chunk(rerank_score: float, text: str = "The model uses attention mechanism."):
    return {
        "text": text,
        "filename": "paper.pdf",
        "page_number": 1,
        "rerank_score": rerank_score,
        "score": 1.0,
    }


# ── hit_rate_at_k ──────────────────────────────────────────────────────────────

class TestHitRateAtK:
    def test_empty_chunks(self):
        assert hit_rate_at_k("attention model", []) == 0.0

    def test_all_chunks_match(self):
        chunks = [
            _chunk(0.8, "attention mechanism in transformers"),
            _chunk(0.7, "the attention head learns patterns"),
        ]
        rate = hit_rate_at_k("attention", chunks)
        assert rate == 1.0

    def test_no_chunks_match(self):
        chunks = [_chunk(0.8, "Convolutional layers for image recognition")]
        rate = hit_rate_at_k("recurrent network lstm gated", chunks)
        assert rate == 0.0

    def test_partial_match(self):
        chunks = [
            _chunk(0.8, "attention is important"),
            _chunk(0.7, "CNN for image tasks only"),
        ]
        rate = hit_rate_at_k("attention", chunks)
        assert rate == 0.5

    def test_k_cap(self):
        chunks = [
            _chunk(0.8, "attention is here"),
            _chunk(0.7, "no match here at all"),
        ]
        rate = hit_rate_at_k("attention", chunks, k=1)
        assert rate == 1.0

    def test_only_stopwords_returns_one(self):
        chunks = [_chunk(0.8)]
        rate = hit_rate_at_k("the a is", chunks)
        assert rate == 1.0


# ── context_precision ─────────────────────────────────────────────────────────

class TestContextPrecision:
    def test_empty(self):
        assert context_precision([]) == 0.0

    def test_all_above_threshold(self):
        chunks = [_chunk(0.8), _chunk(0.9), _chunk(0.7)]
        assert context_precision(chunks, threshold=0.5) == 1.0

    def test_none_above_threshold(self):
        chunks = [_chunk(0.1), _chunk(0.2)]
        assert context_precision(chunks, threshold=0.5) == 0.0

    def test_half_above_threshold(self):
        chunks = [_chunk(0.8), _chunk(0.1)]
        assert context_precision(chunks, threshold=0.5) == 0.5


# ── mean_rerank_score ─────────────────────────────────────────────────────────

class TestMeanRerankScore:
    def test_empty(self):
        assert mean_rerank_score([]) == 0.0

    def test_single(self):
        assert mean_rerank_score([_chunk(0.75)]) == 0.75

    def test_average(self):
        result = mean_rerank_score([_chunk(0.5), _chunk(0.9)])
        assert abs(result - 0.7) < 0.01


# ── rerank_improvement ────────────────────────────────────────────────────────

class TestRerankImprovement:
    def test_empty_inputs(self):
        assert rerank_improvement([], []) == 0.0

    def test_no_rerank_scores_in_raw(self):
        raw = [{"text": "t", "filename": "f", "page_number": 1, "score": 1.0}]
        reranked = [_chunk(0.8)]
        assert rerank_improvement(raw, reranked) == 0.0

    def test_positive_improvement(self):
        raw = [_chunk(0.3), _chunk(0.2)]
        reranked = [_chunk(0.8)]
        delta = rerank_improvement(raw, reranked)
        assert delta > 0

    def test_negative_improvement(self):
        raw = [_chunk(0.9), _chunk(0.85)]
        reranked = [_chunk(0.3)]
        delta = rerank_improvement(raw, reranked)
        assert delta < 0


# ── answer_length ─────────────────────────────────────────────────────────────

class TestAnswerLength:
    def test_empty(self):
        assert answer_length("") == 0

    def test_word_count(self):
        assert answer_length("This is five words here") == 5


# ── groundedness_ratio ────────────────────────────────────────────────────────

class TestGroundednessRatio:
    def test_empty(self):
        assert groundedness_ratio([]) == 0.0

    def test_all_grounded(self):
        chunks = [_chunk(0.8), _chunk(0.9)]
        assert groundedness_ratio(chunks) == 1.0

    def test_none_grounded(self):
        chunks = [_chunk(0.05), _chunk(0.1)]
        assert groundedness_ratio(chunks) == 0.0

    def test_half_grounded(self):
        chunks = [_chunk(0.8), _chunk(0.05)]
        assert groundedness_ratio(chunks) == 0.5


# ── compute_query_metrics ─────────────────────────────────────────────────────

class TestComputeQueryMetrics:
    def test_returns_expected_keys(self):
        chunks = [_chunk(0.8, "attention transformer model architecture")]
        metrics = compute_query_metrics(
            query="What is attention?",
            raw_chunks=chunks,
            reranked_chunks=chunks,
            answer="Attention is a mechanism used in transformers.",
            was_refused=False,
        )
        expected_keys = {
            "query", "hit_rate_at_k", "context_precision", "mean_rerank_score",
            "rerank_improvement", "groundedness_ratio", "answer_length_words",
            "chunks_retrieved", "chunks_after_rerank", "was_refused",
        }
        assert expected_keys.issubset(set(metrics.keys()))

    def test_refused_flag_propagated(self):
        metrics = compute_query_metrics(
            query="q", raw_chunks=[], reranked_chunks=[],
            answer="refused", was_refused=True,
        )
        assert metrics["was_refused"] is True

    def test_chunk_counts(self):
        raw = [_chunk(0.8), _chunk(0.7), _chunk(0.6)]
        reranked = [_chunk(0.8), _chunk(0.7)]
        metrics = compute_query_metrics(
            "query", raw, reranked, "answer", False
        )
        assert metrics["chunks_retrieved"] == 3
        assert metrics["chunks_after_rerank"] == 2


# ── aggregate_session_metrics ─────────────────────────────────────────────────

class TestAggregateSessionMetrics:
    def test_empty(self):
        assert aggregate_session_metrics([]) == {}

    def test_averages_computed(self):
        h1 = compute_query_metrics(
            "q1",
            [_chunk(0.8, "the attention model")],
            [_chunk(0.8, "the attention model")],
            "answer one",
            False,
        )
        h2 = compute_query_metrics(
            "q2",
            [_chunk(0.2)],
            [_chunk(0.2)],
            "answer two words",
            True,
        )
        agg = aggregate_session_metrics([h1, h2])

        assert agg["total_queries"] == 2
        assert agg["fallback_rate"] == 0.5
        assert "avg_hit_rate" in agg
        assert "avg_context_precision" in agg
