"""
Tests for rag/retriever.py and rag/vectorstore.py.

All Cohere API calls are mocked — no real API key needed.
Tests verify FAISS indexing, search, filtering, and persistence.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag.vectorstore import FAISSVectorStore
from rag.retriever import retrieve


# ─── Helpers ──────────────────────────────────────────────────────────────────

DIM = 1024  # matches COHERE_EMBED_DIM


def _random_vectors(n: int, dim: int = DIM) -> np.ndarray:
    rng = np.random.default_rng(seed=42)
    return rng.random((n, dim)).astype(np.float32)


def _make_chunks(n: int, filename: str = "paper.pdf") -> list[dict]:
    return [
        {
            "chunk_id": i,
            "text": f"chunk {i} from {filename}",
            "filename": filename,
            "page_number": (i % 5) + 1,
            "total_pages": 10,
            "file_path": f"/tmp/{filename}",
        }
        for i in range(n)
    ]


# ─── FAISSVectorStore tests ───────────────────────────────────────────────────

class TestFAISSVectorStore:

    def test_add_and_search_returns_results(self):
        store = FAISSVectorStore(embedding_dim=DIM)
        vecs = _random_vectors(5)
        chunks = _make_chunks(5)
        store.add(vecs, chunks)

        query = _random_vectors(1)
        results = store.search(query, top_k=3)

        assert len(results) == 3
        assert "score" in results[0]
        assert "text" in results[0]

    def test_empty_store_returns_empty_list(self):
        store = FAISSVectorStore(embedding_dim=DIM)
        query = _random_vectors(1)
        results = store.search(query, top_k=5)
        assert results == []

    def test_top_k_capped_by_total_vectors(self):
        store = FAISSVectorStore(embedding_dim=DIM)
        store.add(_random_vectors(3), _make_chunks(3))

        results = store.search(_random_vectors(1), top_k=10)

        # Can return at most 3 results (only 3 vectors in the store)
        assert len(results) <= 3

    def test_mismatched_lengths_raises_value_error(self):
        store = FAISSVectorStore(embedding_dim=DIM)
        vecs = _random_vectors(3)
        chunks = _make_chunks(5)  # wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            store.add(vecs, chunks)

    def test_wrong_dim_raises_value_error(self):
        store = FAISSVectorStore(embedding_dim=DIM)
        bad_vecs = _random_vectors(2, dim=512)  # wrong dimension

        with pytest.raises(ValueError, match="shape"):
            store.add(bad_vecs, _make_chunks(2))

    def test_paper_names_reflects_indexed_files(self):
        store = FAISSVectorStore(embedding_dim=DIM)
        store.add(_random_vectors(3), _make_chunks(3, filename="a.pdf"))
        store.add(_random_vectors(2), _make_chunks(2, filename="b.pdf"))

        assert set(store.paper_names) == {"a.pdf", "b.pdf"}

    def test_total_chunks_property(self):
        store = FAISSVectorStore(embedding_dim=DIM)
        store.add(_random_vectors(4), _make_chunks(4))
        assert store.total_chunks == 4

    def test_remove_paper_updates_chunks(self):
        store = FAISSVectorStore(embedding_dim=DIM)
        store.add(_random_vectors(3), _make_chunks(3, filename="keep.pdf"))
        store.add(_random_vectors(2), _make_chunks(2, filename="remove.pdf"))

        store.remove_paper("remove.pdf")

        assert "remove.pdf" not in store.paper_names
        assert store.total_chunks == 3

    def test_save_and_load_roundtrip(self, tmp_path):
        store = FAISSVectorStore(embedding_dim=DIM)
        vecs = _random_vectors(4)
        store.add(vecs, _make_chunks(4))

        store.save(str(tmp_path))
        loaded = FAISSVectorStore.load(str(tmp_path))

        assert loaded.total_chunks == 4
        assert loaded.index.ntotal == 4

    def test_load_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FAISSVectorStore.load(str(tmp_path))

    def test_results_sorted_best_first(self):
        """Nearest vector should appear first (lowest L2 distance)."""
        store = FAISSVectorStore(embedding_dim=DIM)
        rng = np.random.default_rng(seed=0)
        base = rng.random((1, DIM)).astype(np.float32)

        # Create a vector very close to `base` (first chunk)
        close = base + rng.random((1, DIM)).astype(np.float32) * 0.01
        far = rng.random((1, DIM)).astype(np.float32) * 10.0

        vecs = np.vstack([close, far])
        store.add(vecs, _make_chunks(2))

        results = store.search(base, top_k=2)
        assert results[0]["score"] <= results[1]["score"]


# ─── retrieve() tests ─────────────────────────────────────────────────────────

class TestRetrieve:

    @patch("rag.retriever.embed_query")
    def test_retrieve_calls_store_search(self, mock_embed):
        mock_embed.return_value = _random_vectors(1)
        store = FAISSVectorStore(embedding_dim=DIM)
        store.add(_random_vectors(5), _make_chunks(5))

        results = retrieve("what is attention?", store, top_k=3)

        mock_embed.assert_called_once_with("what is attention?")
        assert len(results) <= 3

    @patch("rag.retriever.embed_query")
    def test_paper_filter_restricts_results(self, mock_embed):
        mock_embed.return_value = _random_vectors(1)
        store = FAISSVectorStore(embedding_dim=DIM)
        store.add(_random_vectors(5), _make_chunks(5, filename="a.pdf"))
        store.add(_random_vectors(5), _make_chunks(5, filename="b.pdf"))

        results = retrieve("some query", store, top_k=10, paper_filter="a.pdf")

        assert all(r["filename"] == "a.pdf" for r in results)

    @patch("rag.retriever.embed_query")
    def test_empty_store_returns_empty_list(self, mock_embed):
        mock_embed.return_value = _random_vectors(1)
        store = FAISSVectorStore(embedding_dim=DIM)

        results = retrieve("anything", store)

        assert results == []
