"""
End-to-end chat flow tests.

All external API calls (Cohere Embed, Cohere Rerank, Groq) are mocked
so the suite runs in CI without real credentials.

Tests verify:
  • The RAG pipeline wiring (retrieve → rerank → generate).
  • Citation formatting.
  • Reranker score propagation.
  • Edge cases (no chunks, empty store).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.vectorstore import FAISSVectorStore
from rag.citations import (
    format_citations,
    format_source_snippets,
    build_apa_citation,
    build_ieee_citation,
    _rerank_label,
    _l2_label,
)


# ─── Constants ────────────────────────────────────────────────────────────────

DIM = 1024


def _rnd(n: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed=7)
    return rng.random((n, DIM)).astype(np.float32)


def _chunk(i: int = 0, filename: str = "paper.pdf", page: int = 1) -> dict:
    return {
        "chunk_id": i,
        "text": f"This is chunk {i} about neural networks and transformers.",
        "filename": filename,
        "page_number": page,
        "total_pages": 10,
        "file_path": f"/tmp/{filename}",
        "vector_id": i,
        "score": 0.5,
    }


# ─── Citation tests ───────────────────────────────────────────────────────────

class TestFormatCitations:

    def test_empty_chunks_returns_no_sources_message(self):
        result = format_citations([])
        assert "No sources" in result

    def test_single_chunk_produces_one_citation(self):
        chunks = [_chunk(0, page=3)]
        result = format_citations(chunks)
        assert "paper.pdf" in result
        assert "Page 3" in result

    def test_deduplicates_same_page(self):
        chunks = [
            _chunk(0, page=2),
            _chunk(1, page=2),  # same file, same page
        ]
        result = format_citations(chunks)
        assert result.count("Page 2") == 1

    def test_multiple_distinct_pages_all_listed(self):
        chunks = [_chunk(0, page=1), _chunk(1, page=4), _chunk(2, page=7)]
        result = format_citations(chunks)
        assert "Page 1" in result
        assert "Page 4" in result
        assert "Page 7" in result

    def test_rerank_score_shown_when_present(self):
        chunk = _chunk()
        chunk["rerank_score"] = 0.85
        result = format_citations([chunk])
        assert "0.85" in result or "High" in result


class TestFormatSourceSnippets:

    def test_empty_returns_empty_string(self):
        assert format_source_snippets([]) == ""

    def test_snippet_truncated_at_max_chars(self):
        chunk = _chunk()
        chunk["text"] = "x" * 500
        result = format_source_snippets([chunk], max_chars=100)
        assert "…" in result

    def test_short_text_not_truncated(self):
        chunk = _chunk()
        chunk["text"] = "Short text."
        result = format_source_snippets([chunk], max_chars=500)
        assert "…" not in result

    def test_filename_and_page_in_output(self):
        chunk = _chunk(filename="myreview.pdf", page=6)
        result = format_source_snippets([chunk])
        assert "myreview.pdf" in result
        assert "Page 6" in result


class TestAcademicCitations:

    def test_apa_format(self):
        result = build_apa_citation("attention_is_all_you_need.pdf", 3)
        assert "Page 3" in result
        assert "." in result  # ends with period

    def test_ieee_format(self):
        result = build_ieee_citation("bert_paper.pdf", 7, 2)
        assert "[2]" in result
        assert "p. 7" in result

    def test_filename_cleaned_in_apa(self):
        result = build_apa_citation("my_cool-paper.pdf", 1)
        assert "_" not in result
        assert "-" not in result

    def test_rerank_label_high(self):
        assert "High" in _rerank_label(0.9)

    def test_rerank_label_medium(self):
        assert "Medium" in _rerank_label(0.5)

    def test_rerank_label_low(self):
        assert "Low" in _rerank_label(0.1)

    def test_l2_label_high(self):
        assert _l2_label(0.1) == "High"

    def test_l2_label_medium(self):
        assert _l2_label(1.0) == "Medium"

    def test_l2_label_low(self):
        assert _l2_label(2.0) == "Low"


# ─── End-to-end pipeline tests (mocked APIs) ─────────────────────────────────

class TestEndToEndPipeline:

    def _populate_store(self) -> FAISSVectorStore:
        store = FAISSVectorStore(embedding_dim=DIM)
        chunks = [_chunk(i, page=i + 1) for i in range(5)]
        store.add(_rnd(5), chunks)
        return store

    @patch("rag.retriever.embed_query")
    def test_retrieve_returns_chunks(self, mock_embed):
        mock_embed.return_value = _rnd(1)
        from rag.retriever import retrieve

        store = self._populate_store()
        results = retrieve("attention mechanism", store, top_k=3)

        assert len(results) == 3
        assert all("text" in r for r in results)

    @patch("rag.reranker.cohere")
    def test_rerank_adds_rerank_score(self, mock_cohere_module):
        """Verify that rerank() attaches rerank_score to each chunk."""
        from rag.reranker import rerank

        # Build mock Cohere response
        mock_result = MagicMock()
        mock_result.index = 0
        mock_result.relevance_score = 0.92

        mock_response = MagicMock()
        mock_response.results = [mock_result]

        mock_client = MagicMock()
        mock_client.rerank.return_value = mock_response
        mock_cohere_module.Client.return_value = mock_client

        chunks = [_chunk(0)]

        # Patch the module-level _client so it uses our mock
        import rag.reranker as reranker_mod
        original_client = reranker_mod._client
        reranker_mod._client = mock_client

        try:
            result = rerank("query", chunks, top_n=1)
        finally:
            reranker_mod._client = original_client

        assert len(result) == 1
        assert result[0]["rerank_score"] == 0.92

    @patch("rag.generator.Groq")
    def test_generate_answer_returns_string(self, mock_groq_cls):
        """Verify generate_answer returns a non-empty string."""
        from rag.generator import generate_answer

        mock_message = MagicMock()
        mock_message.content = "The attention mechanism works by computing queries, keys, and values."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq_cls.return_value = mock_client

        import rag.generator as gen_mod
        original_client = gen_mod._client
        gen_mod._client = mock_client

        try:
            answer, used = generate_answer("What is attention?", [_chunk(0)])
        finally:
            gen_mod._client = original_client

        assert isinstance(answer, str)
        assert len(answer) > 0
        assert len(used) == 1

    @patch("rag.generator.Groq")
    def test_generate_answer_with_no_chunks_returns_fallback(self, mock_groq_cls):
        from rag.generator import generate_answer

        answer, used = generate_answer("anything", [])

        assert "could not find" in answer.lower()
        assert used == []
