"""
Tests for the streaming generator path (stream_answer).
All Groq API calls are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch
from rag.generator import stream_answer, generate_answer, build_context


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _chunk(text: str = "Attention is a key mechanism.", page: int = 1):
    return {
        "text": text,
        "filename": "paper.pdf",
        "page_number": page,
        "rerank_score": 0.8,
    }


# ── build_context ──────────────────────────────────────────────────────────────

class TestBuildContext:
    def test_single_chunk(self):
        ctx = build_context([_chunk()])
        assert "paper.pdf" in ctx
        assert "Attention" in ctx

    def test_multiple_chunks_numbered(self):
        ctx = build_context([_chunk("First chunk."), _chunk("Second chunk.", 2)])
        assert "[1]" in ctx
        assert "[2]" in ctx


# ── stream_answer ──────────────────────────────────────────────────────────────

class TestStreamAnswer:
    def test_empty_chunks_yields_fallback(self):
        tokens = list(stream_answer("query", []))
        full = "".join(tokens)
        assert "could not find" in full.lower()

    def test_streaming_yields_tokens(self, mocker):
        mock_client = MagicMock()
        mocker.patch(
            "rag.generator.get_groq_client", return_value=mock_client
        )

        # Simulate the context manager returned by client.chat.completions.stream
        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_stream_ctx.text_stream = iter(["Hello", " world", "."])
        mock_client.chat.completions.stream.return_value = mock_stream_ctx

        tokens = list(stream_answer("What is attention?", [_chunk()]))
        assert tokens == ["Hello", " world", "."]

    def test_streaming_full_text(self, mocker):
        mock_client = MagicMock()
        mocker.patch("rag.generator.get_groq_client", return_value=mock_client)

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__enter__ = MagicMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__exit__ = MagicMock(return_value=False)
        mock_stream_ctx.text_stream = iter(["The answer", " is here."])
        mock_client.chat.completions.stream.return_value = mock_stream_ctx

        text = "".join(stream_answer("query", [_chunk()]))
        assert text == "The answer is here."


# ── generate_answer (existing, verify still works) ────────────────────────────

class TestGenerateAnswerStillWorks:
    def test_empty_chunks_fallback(self):
        answer, chunks = generate_answer("query", [])
        assert "could not find" in answer.lower()
        assert chunks == []

    def test_returns_answer_and_chunks(self, mocker):
        mock_client = MagicMock()
        mocker.patch("rag.generator.get_groq_client", return_value=mock_client)

        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Generated answer text."
        mock_client.chat.completions.create.return_value = mock_response

        answer, used = generate_answer("query", [_chunk()])
        assert answer == "Generated answer text."
        assert len(used) == 1
