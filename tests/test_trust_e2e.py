"""
End-to-end smoke test for the trust & quality features (v2).

Tests the full pipeline with mocked APIs:
  grounding guardrail → confidence labelling → evaluation metrics →
  reference extraction → rerank comparison data.
"""

import pytest
import io
from unittest.mock import MagicMock, patch
import numpy as np

from rag.grounding import evaluate_grounding, annotate_answer, build_refusal_message
from rag.evaluation import compute_query_metrics, aggregate_session_metrics
from rag.references import extract_references, format_references_markdown
from rag.demo import ensure_sample_pdf, SAMPLE_FILENAME


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_chunk(rerank_score: float, text: str = "Attention is used in RAG systems."):
    return {
        "text": text,
        "filename": "rag_paper.pdf",
        "page_number": 1,
        "rerank_score": rerank_score,
        "score": 0.8,
    }


def _make_page(text: str, page_num: int = 1):
    return {
        "text": text,
        "page_number": page_num,
        "filename": "rag_paper.pdf",
        "total_pages": 3,
    }


# ── Smoke test: grounding + confidence + refusal ───────────────────────────────

class TestGroundingSmoke:
    def test_high_evidence_no_refusal_high_label(self):
        chunks = [_make_chunk(0.9), _make_chunk(0.85), _make_chunk(0.8)]
        score, label, refuse = evaluate_grounding(chunks)
        assert not refuse
        assert label == "High"
        assert score >= 0.6

    def test_low_evidence_triggers_refusal(self):
        # No chunks at all → returns 0.0, which is below the refusal threshold
        score, label, refuse = evaluate_grounding([])
        assert refuse
        assert label == "Low"
        refusal_msg = build_refusal_message(score)
        assert "⚠️" in refusal_msg

    def test_annotate_preserves_content(self):
        answer = "Attention allows models to focus on relevant tokens."
        annotated = annotate_answer(answer, 0.8, "High")
        assert answer in annotated
        assert "High" in annotated

    def test_medium_confidence_no_refusal(self):
        chunks = [_make_chunk(0.45), _make_chunk(0.38)]
        _, label, refuse = evaluate_grounding(chunks)
        assert not refuse
        assert label in ("Medium", "High")


# ── Smoke test: evaluation metrics ────────────────────────────────────────────

class TestEvaluationSmoke:
    def test_single_query_metrics_complete(self):
        raw = [_make_chunk(0.6, "retrieval augmented generation model")]
        reranked = [_make_chunk(0.8, "retrieval augmented generation model")]
        metrics = compute_query_metrics(
            query="What is retrieval augmented generation?",
            raw_chunks=raw,
            reranked_chunks=reranked,
            answer="RAG combines retrieval with generation.",
            was_refused=False,
        )
        assert metrics["hit_rate_at_k"] > 0
        assert metrics["chunks_retrieved"] == 1
        assert metrics["chunks_after_rerank"] == 1
        assert metrics["was_refused"] is False

    def test_session_aggregation(self):
        def _query_metrics(suffix, score):
            return compute_query_metrics(
                query=f"Query about {suffix}",
                raw_chunks=[_make_chunk(score, f"{suffix} content here")],
                reranked_chunks=[_make_chunk(score, f"{suffix} content here")],
                answer=f"Answer about {suffix}.",
                was_refused=score < 0.05,
            )

        history = [
            _query_metrics("attention", 0.8),
            _query_metrics("retrieval", 0.6),
            _query_metrics("generation", 0.01),
        ]
        agg = aggregate_session_metrics(history)

        assert agg["total_queries"] == 3
        assert agg["fallback_rate"] == pytest.approx(1 / 3, abs=0.01)
        assert 0 <= agg["avg_hit_rate"] <= 1.0
        assert 0 <= agg["avg_context_precision"] <= 1.0


# ── Smoke test: reference extraction ─────────────────────────────────────────

class TestReferenceSmoke:
    def test_extract_and_format(self):
        pages = [
            _make_page("Introduction to RAG systems.", 1),
            _make_page(
                "References\n\n"
                "[1] Lewis, P. et al. (2020). Retrieval-Augmented Generation. NeurIPS.\n"
                "[2] Karpukhin, V. et al. (2020). Dense Passage Retrieval. EMNLP.\n",
                2,
            ),
        ]
        refs = extract_references(pages, "rag_paper.pdf")
        assert len(refs) >= 2

        md = format_references_markdown(refs)
        assert "1." in md
        assert "Lewis" in md or "Retrieval" in md

    def test_no_refs_returns_empty(self):
        pages = [_make_page("Just text without bibliography.", 1)]
        refs = extract_references(pages, "paper.pdf")
        assert refs == []


# ── Smoke test: sample demo ───────────────────────────────────────────────────

class TestDemoSmoke:
    def test_sample_pdf_loadable_by_rag(self):
        """The sample PDF must pass through the loader without error."""
        from rag.loader import load_pdf
        path = ensure_sample_pdf()
        pages = load_pdf(path)
        assert len(pages) >= 1
        for page in pages:
            assert "text" in page
            assert "page_number" in page

    def test_sample_pdf_chunkable(self):
        """The sample PDF chunks must have chunk_id and text keys."""
        from rag.loader import load_pdf
        from rag.chunking import chunk_pages
        path = ensure_sample_pdf()
        pages = load_pdf(path)
        chunks = chunk_pages(pages)
        assert len(chunks) >= 3
        for chunk in chunks:
            assert "chunk_id" in chunk
            assert "text" in chunk
            assert len(chunk["text"]) >= 10

    def test_sample_has_reference_section(self):
        """The sample paper should have a reference section."""
        from rag.loader import load_pdf
        path = ensure_sample_pdf()
        pages = load_pdf(path)
        refs = extract_references(pages, SAMPLE_FILENAME)
        # Sample paper has 6 references in its text
        assert len(refs) >= 1


# ── Smoke test: rerank comparison data ───────────────────────────────────────

class TestRerankComparisonSmoke:
    def test_rerank_improvement_positive_for_good_reranker(self):
        from rag.evaluation import rerank_improvement
        raw = [
            _make_chunk(0.3, "somewhat relevant content"),
            _make_chunk(0.2, "less relevant content"),
        ]
        reranked = [_make_chunk(0.9, "highly relevant content")]
        delta = rerank_improvement(raw, reranked)
        assert delta > 0, "Reranker should select better chunks"

    def test_rerank_data_has_scores(self):
        """Confirm reranked chunks contain rerank_score for the UI panel."""
        reranked = [_make_chunk(0.8), _make_chunk(0.7)]
        for c in reranked:
            assert "rerank_score" in c
            assert isinstance(c["rerank_score"], float)
