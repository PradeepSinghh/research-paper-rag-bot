"""
FAISS vector store — build, persist, query, and manage the index.

Uses IndexFlatL2 (exact nearest-neighbour search) which is perfectly
adequate for the typical research session (< 100 k vectors).  For
larger corpora, swap to IndexIVFFlat by changing _build_index().

The store keeps a parallel Python list (self.chunks) that mirrors the
FAISS vectors so we can retrieve full chunk metadata after a search.
"""

import os
import pickle
from typing import List, Dict, Any, Optional

import numpy as np
import faiss

from utils.logger import get_logger
from utils.file_utils import ensure_dir
from utils import config

logger = get_logger(__name__)

# File names used when persisting to disk
_INDEX_FILE = "faiss.index"
_CHUNKS_FILE = "chunks.pkl"


class FAISSVectorStore:
    """
    Wraps a FAISS index together with chunk metadata.

    Public API
    ----------
    add(embeddings, chunks)  — add new vectors + metadata
    search(query_emb, top_k) — nearest-neighbour search
    remove_paper(filename)   — drop all chunks for one paper
    save(index_dir)          — persist to disk
    load(index_dir)          — class method; restore from disk
    total_chunks             — property; current chunk count
    paper_names              — property; list of indexed filenames
    """

    def __init__(self, embedding_dim: int = config.COHERE_EMBED_DIM) -> None:
        self.embedding_dim = embedding_dim
        self.index: faiss.IndexFlatL2 = faiss.IndexFlatL2(embedding_dim)
        # Parallel list: self.chunks[i] corresponds to FAISS vector i
        self.chunks: List[Dict[str, Any]] = []

    # ── Adding vectors ─────────────────────────────────────────────────────────

    def add(
        self,
        embeddings: np.ndarray,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """
        Add *embeddings* and their corresponding *chunks* to the store.

        Args:
            embeddings: Float32 array of shape (n, embedding_dim).
            chunks:     List of chunk dicts (same length as embeddings).

        Raises:
            ValueError: Mismatched lengths or wrong embedding dimension.
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Length mismatch: {len(embeddings)} embeddings vs "
                f"{len(chunks)} chunks."
            )
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected shape (n, {self.embedding_dim}), "
                f"got {embeddings.shape}."
            )

        base_id = len(self.chunks)

        # Assign a stable vector_id so retrieval can map back to metadata
        for i, chunk in enumerate(chunks):
            stamped = dict(chunk)
            stamped["vector_id"] = base_id + i
            self.chunks.append(stamped)

        # FAISS requires contiguous float32
        vectors = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.add(vectors)

        logger.info(
            f"Added {len(embeddings)} vector(s). "
            f"Store total: {self.index.ntotal}"
        )

    # ── Querying ───────────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = config.TOP_K_RETRIEVE,
    ) -> List[Dict[str, Any]]:
        """
        Return the *top_k* most similar chunks (ascending L2 distance).

        Each returned dict is a copy of the stored chunk dict with an
        additional ``score`` key (L2 distance — lower means more similar).

        Args:
            query_embedding: Float32 array of shape (1, embedding_dim).
            top_k:           Maximum number of results to return.

        Returns:
            List of chunk dicts, sorted best-first (lowest L2 distance).
        """
        if self.index.ntotal == 0:
            logger.warning("Vector store is empty — no documents indexed yet.")
            return []

        k = min(top_k, self.index.ntotal)
        query = np.ascontiguousarray(query_embedding, dtype=np.float32)
        distances, indices = self.index.search(query, k)

        results: List[Dict[str, Any]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue  # FAISS pads with -1 when fewer than k vectors exist
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(dist)
            results.append(chunk)

        return results

    # ── Paper management ───────────────────────────────────────────────────────

    def remove_paper(self, filename: str) -> None:
        """
        Remove all chunks that belong to *filename*.

        Because FAISS does not support in-place deletion, this method
        removes the metadata entries from self.chunks.  The stale FAISS
        vectors remain but will never be returned (they are filtered by
        paper_filter in the retriever).  Call save() + load() to rebuild
        a clean index if storage efficiency matters.

        Args:
            filename: Basename of the PDF to remove (e.g., "paper.pdf").
        """
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.get("filename") != filename]
        removed = before - len(self.chunks)
        logger.info(f"Removed {removed} chunk(s) for '{filename}' from metadata.")

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, index_dir: str) -> None:
        """
        Persist the FAISS index and chunk metadata to *index_dir*.

        Args:
            index_dir: Directory path (created if it does not exist).
        """
        ensure_dir(index_dir)
        faiss.write_index(self.index, os.path.join(index_dir, _INDEX_FILE))
        with open(os.path.join(index_dir, _CHUNKS_FILE), "wb") as fh:
            pickle.dump(self.chunks, fh)
        logger.info(f"Vector store saved to '{index_dir}'")

    @classmethod
    def load(cls, index_dir: str) -> "FAISSVectorStore":
        """
        Restore a previously saved vector store from *index_dir*.

        Args:
            index_dir: Directory containing faiss.index and chunks.pkl.

        Returns:
            Populated FAISSVectorStore instance.

        Raises:
            FileNotFoundError: If the expected files are not present.
        """
        index_path = os.path.join(index_dir, _INDEX_FILE)
        chunks_path = os.path.join(index_dir, _CHUNKS_FILE)

        if not os.path.exists(index_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"No saved index found in '{index_dir}'. "
                "Upload and process documents first."
            )

        store = cls()
        store.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as fh:
            store.chunks = pickle.load(fh)

        logger.info(
            f"Loaded index: {store.index.ntotal} vector(s), "
            f"{len(store.chunks)} chunk(s)."
        )
        return store

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def total_chunks(self) -> int:
        """Number of chunks currently in the store."""
        return len(self.chunks)

    @property
    def paper_names(self) -> List[str]:
        """Sorted list of unique filenames currently indexed."""
        return sorted({c["filename"] for c in self.chunks})
