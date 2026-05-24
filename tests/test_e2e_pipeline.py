"""
End-to-end pipeline test with a realistic dummy research paper.

Creates a multi-page academic-style PDF in-memory, then runs the
COMPLETE pipeline:

    load → chunk → embed* → FAISS index → retrieve* → rerank* → generate* → cite

(*) Cohere Embed, Cohere Rerank, and Groq are mocked with realistic
    return values so the suite runs without real API keys.

Run with:
    pytest tests/test_e2e_pipeline.py -v -s
"""

import os
import textwrap
from typing import List
from unittest.mock import MagicMock, patch

import fitz  # PyMuPDF
import numpy as np
import pytest

# ── Project imports ────────────────────────────────────────────────────────────
from rag.loader import load_pdf
from rag.chunking import chunk_pages
from rag.vectorstore import FAISSVectorStore
from rag.retriever import retrieve
from rag.reranker import rerank
from rag.generator import generate_answer
from rag.citations import format_citations, format_source_snippets, build_apa_citation
from rag.compare import compare_papers

DIM = 1024   # Cohere embed-english-v3.0 output dimension
RNG = np.random.default_rng(seed=99)


# ══════════════════════════════════════════════════════════════════════════════
# Dummy paper content
# ══════════════════════════════════════════════════════════════════════════════

_PAPER_PAGES = [
    # Page 1 — Title & Abstract
    textwrap.dedent("""\
        Attention-Augmented Transformers for Scientific Document Understanding

        Authors: A. Researcher, B. Scientist, C. Engineer
        Institution: Institute of Artificial Intelligence Research
        Year: 2024

        Abstract

        We present AttentionRAG, a novel architecture that combines multi-head
        self-attention with retrieval-augmented generation for scientific document
        understanding. Our model achieves state-of-the-art performance on the
        SciQA and PubMedQA benchmarks, surpassing GPT-4 by 7.3% on exact-match
        accuracy. The key innovation is a cross-document attention mechanism that
        allows the model to simultaneously reason over multiple scientific papers.
        We release the model weights and dataset under an open-source license.
    """),

    # Page 2 — Introduction
    textwrap.dedent("""\
        1. Introduction

        Scientific document understanding is a challenging NLP task that requires
        integrating knowledge across multiple sources, understanding domain-specific
        terminology, and reasoning over complex claims.

        Prior work on retrieval-augmented generation (RAG) systems has shown
        promising results for open-domain question answering. However, existing
        systems struggle with scientific texts for three key reasons:

        (1) Scientific papers are long and densely structured, making chunking
            critical for effective retrieval.
        (2) Domain-specific embeddings trained on general web text fail to
            capture the nuanced semantics of scientific language.
        (3) Standard reranking models are not calibrated for scientific relevance.

        Our contributions are:
        - AttentionRAG: a retrieval-augmented transformer tuned for science.
        - SciChunk: an adaptive chunking strategy preserving section boundaries.
        - A new benchmark: SciQA-2024 with 12,000 expert-annotated QA pairs.
    """),

    # Page 3 — Methodology
    textwrap.dedent("""\
        2. Methodology

        2.1 Document Ingestion
        PDFs are parsed using a hybrid layout-aware extractor that preserves
        section headings, figure captions, and table structure. Text is cleaned
        to remove header/footer artifacts and hyphenation errors.

        2.2 Adaptive Chunking (SciChunk)
        Unlike fixed-size chunking, SciChunk detects section boundaries using
        a lightweight CNN classifier trained on 50,000 scientific papers.
        Chunks are formed at natural section transitions with a 10% overlap to
        prevent context loss at boundaries. Mean chunk length: 412 tokens.

        2.3 Embeddings
        We fine-tune a 110M parameter encoder on the S2ORC dataset (Lo et al.,
        2020) consisting of 81 million scientific papers. The encoder produces
        1024-dimensional representations that capture scientific semantics
        significantly better than general-purpose encoders on the MTEB benchmark.

        2.4 Reranking
        Retrieved candidates are reranked using a cross-encoder model trained
        on (query, scientific passage) pairs from the PubMed and arXiv datasets.
        The cross-encoder jointly encodes the query and passage, enabling precise
        relevance scoring that bi-encoder retrieval alone cannot achieve.
    """),

    # Page 4 — Experiments
    textwrap.dedent("""\
        3. Experiments and Results

        3.1 Datasets
        We evaluate on three benchmarks:
        - SciQA-2024: 12,000 scientific QA pairs (our contribution).
        - PubMedQA: 1,000 biomedical yes/no questions (Jin et al., 2019).
        - QASPER: 5,049 NLP paper QA pairs (Dasigi et al., 2021).

        3.2 Baselines
        We compare against:
        - BM25 + GPT-4: standard sparse retrieval with LLM generation.
        - DPR + LLaMA-2: dense passage retrieval with open-source LLM.
        - ColBERT + Claude-3: late-interaction retrieval with frontier LLM.

        3.3 Main Results
        AttentionRAG achieves the following exact-match accuracy scores:
        - SciQA-2024:  AttentionRAG 78.4%  vs  GPT-4 71.1%  (+7.3%)
        - PubMedQA:    AttentionRAG 82.1%  vs  GPT-4 79.5%  (+2.6%)
        - QASPER:      AttentionRAG 54.3%  vs  GPT-4 51.8%  (+2.5%)

        Our adaptive chunking strategy alone accounts for 3.1% of the gain,
        confirming that chunk quality is a first-class concern in RAG pipelines.
    """),

    # Page 5 — Limitations & Conclusion
    textwrap.dedent("""\
        4. Limitations

        Despite strong results, AttentionRAG has the following limitations:

        (1) Computational cost: the cross-encoder reranker adds 200ms latency
            per query on a single A100 GPU. Real-time applications may need
            a lighter reranking model.
        (2) Language coverage: the model was trained exclusively on English
            scientific papers. Performance on non-English documents is unknown.
        (3) Multi-modal content: figures, equations, and tables are currently
            ignored during ingestion. Including these modalities could further
            improve performance.
        (4) Hallucination: while retrieval grounding significantly reduces
            hallucination, the model occasionally generates plausible-sounding
            but incorrect citations, especially for obscure topics.

        5. Conclusion

        We presented AttentionRAG, a retrieval-augmented transformer for
        scientific document understanding. Through adaptive chunking, scientific
        embeddings, and precision reranking, we achieve consistent gains over
        strong baselines across three benchmarks. Future work will extend the
        system to multi-modal inputs and non-English scientific corpora.

        Acknowledgements: Funded by NSF Grant #2024-AI-9812.

        References
        [1] Lo et al. (2020) S2ORC: The Semantic Scholar Open Research Corpus.
        [2] Jin et al. (2019) PubMedQA: A Dataset for Biomedical Research QA.
        [3] Dasigi et al. (2021) A Dataset of Information-Seeking Questions from NLP Papers.
    """),
]


# ══════════════════════════════════════════════════════════════════════════════
# PDF factory
# ══════════════════════════════════════════════════════════════════════════════

def _create_dummy_paper(path: str, title: str = "attention_rag.pdf") -> str:
    """
    Write a realistic multi-page academic PDF to *path* and return *path*.
    """
    doc = fitz.open()
    for page_text in _PAPER_PAGES:
        page = doc.new_page(width=595, height=842)  # A4
        # Insert page text with word wrapping via text box
        rect = fitz.Rect(50, 60, 545, 800)
        page.insert_textbox(rect, page_text, fontsize=10, align=fitz.TEXT_ALIGN_LEFT)
    doc.save(path)
    doc.close()
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Mock factories
# ══════════════════════════════════════════════════════════════════════════════

def _mock_cohere_embed_response(texts: List[str]) -> MagicMock:
    """Return a Cohere embed response with random 1024-dim vectors."""
    resp = MagicMock()
    resp.embeddings = RNG.random((len(texts), DIM)).astype(np.float32).tolist()
    return resp


def _mock_cohere_rerank_response(chunks: List[dict], top_n: int) -> MagicMock:
    """Return a Cohere rerank response that preserves original order."""
    resp = MagicMock()
    results = []
    for i in range(min(top_n, len(chunks))):
        r = MagicMock()
        r.index = i
        r.relevance_score = round(0.95 - i * 0.08, 3)   # decreasing scores
        results.append(r)
    resp.results = results
    return resp


def _mock_groq_response(answer_text: str) -> MagicMock:
    """Return a Groq chat completion response."""
    msg = MagicMock()
    msg.content = answer_text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def dummy_pdf(tmp_path_factory) -> str:
    """Create the dummy research paper PDF once for the whole module."""
    pdf_dir = tmp_path_factory.mktemp("papers")
    pdf_path = str(pdf_dir / "attention_rag.pdf")
    _create_dummy_paper(pdf_path)
    return pdf_path


@pytest.fixture(scope="module")
def loaded_pages(dummy_pdf):
    """Load pages from the dummy PDF."""
    return load_pdf(dummy_pdf)


@pytest.fixture(scope="module")
def chunks(loaded_pages):
    """Chunk the loaded pages."""
    return chunk_pages(loaded_pages, chunk_size=400, chunk_overlap=60)


@pytest.fixture(scope="module")
def populated_store(chunks):
    """Build a FAISS store from embedded chunks (Cohere embed mocked)."""
    store = FAISSVectorStore(embedding_dim=DIM)
    texts = [c["text"] for c in chunks]

    # Patch Cohere client at the module level
    import rag.embeddings as emb_mod
    mock_client = MagicMock()
    mock_client.embed.side_effect = lambda **kw: _mock_cohere_embed_response(kw["texts"])
    original = emb_mod._client
    emb_mod._client = mock_client

    embeddings = __import__("rag.embeddings", fromlist=["embed_documents"]).embed_documents(texts)
    store.add(embeddings, chunks)

    emb_mod._client = original
    return store, mock_client


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — PDF Loading
# ══════════════════════════════════════════════════════════════════════════════

class TestStep1_PDFLoading:

    def test_loads_all_five_pages(self, loaded_pages, capsys):
        print("\n" + "═" * 60)
        print("STEP 1 — PDF LOADING")
        print("═" * 60)
        print(f"  Pages extracted : {len(loaded_pages)}")
        for p in loaded_pages:
            preview = p["text"][:80].replace("\n", " ")
            print(f"  Page {p['page_number']:>2}: {preview}…")
        assert len(loaded_pages) == 5

    def test_page_metadata_complete(self, loaded_pages):
        for page in loaded_pages:
            assert page["filename"] == "attention_rag.pdf"
            assert page["total_pages"] == 5
            assert "text" in page and len(page["text"]) > 50

    def test_content_extracted_correctly(self, loaded_pages):
        full_text = " ".join(p["text"] for p in loaded_pages)
        assert "AttentionRAG" in full_text
        assert "Methodology" in full_text
        assert "Limitations" in full_text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Chunking
# ══════════════════════════════════════════════════════════════════════════════

class TestStep2_Chunking:

    def test_produces_multiple_chunks(self, chunks, capsys):
        print("\n" + "═" * 60)
        print("STEP 2 — CHUNKING")
        print("═" * 60)
        print(f"  Total chunks    : {len(chunks)}")
        sizes = [len(c["text"]) for c in chunks]
        print(f"  Min chunk size  : {min(sizes)} chars")
        print(f"  Max chunk size  : {max(sizes)} chars")
        print(f"  Mean chunk size : {sum(sizes)//len(sizes)} chars")
        for i, c in enumerate(chunks[:3]):
            print(f"\n  Chunk {c['chunk_id']} (page {c['page_number']}):")
            print(f"    {c['text'][:120].replace(chr(10), ' ')}…")
        assert len(chunks) >= 5

    def test_chunks_have_metadata(self, chunks):
        for c in chunks:
            assert "chunk_id" in c
            assert "filename" in c
            assert "page_number" in c
            assert c["filename"] == "attention_rag.pdf"

    def test_chunk_ids_sequential(self, chunks):
        assert [c["chunk_id"] for c in chunks] == list(range(len(chunks)))

    def test_chunks_cover_all_sections(self, chunks):
        all_text = " ".join(c["text"] for c in chunks)
        for keyword in ["Introduction", "Methodology", "Results", "Limitations"]:
            assert keyword in all_text, f"'{keyword}' not found in any chunk"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Embedding + Indexing
# ══════════════════════════════════════════════════════════════════════════════

class TestStep3_EmbeddingAndIndexing:

    def test_store_has_correct_vector_count(self, populated_store, chunks, capsys):
        store, _ = populated_store
        print("\n" + "═" * 60)
        print("STEP 3 — EMBEDDING & FAISS INDEXING")
        print("═" * 60)
        print(f"  Vectors indexed : {store.index.ntotal}")
        print(f"  Chunk count     : {store.total_chunks}")
        print(f"  Papers indexed  : {store.paper_names}")
        assert store.index.ntotal == len(chunks)

    def test_paper_names_correct(self, populated_store):
        store, _ = populated_store
        assert store.paper_names == ["attention_rag.pdf"]

    def test_save_and_reload_roundtrip(self, populated_store, tmp_path, capsys):
        store, _ = populated_store
        store.save(str(tmp_path))
        reloaded = FAISSVectorStore.load(str(tmp_path))
        print(f"\n  [Save/Load] Reloaded {reloaded.total_chunks} chunks from disk ✓")
        assert reloaded.total_chunks == store.total_chunks
        assert reloaded.index.ntotal == store.index.ntotal


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Retrieval
# ══════════════════════════════════════════════════════════════════════════════

class TestStep4_Retrieval:

    @patch("rag.retriever.embed_query")
    def test_retrieves_top_k_chunks(self, mock_embed, populated_store, capsys):
        store, _ = populated_store
        mock_embed.return_value = RNG.random((1, DIM)).astype(np.float32)

        query = "What methodology does AttentionRAG use for chunking?"
        results = retrieve(query, store, top_k=5)

        print("\n" + "═" * 60)
        print("STEP 4 — RETRIEVAL (FAISS nearest-neighbour)")
        print("═" * 60)
        print(f"  Query  : {query}")
        print(f"  Hits   : {len(results)}")
        for i, r in enumerate(results):
            print(f"  [{i+1}] page={r['page_number']} score={r['score']:.4f}  "
                  f"{r['text'][:80].replace(chr(10),' ')}…")

        assert len(results) == 5
        assert all("score" in r for r in results)
        assert all(r["filename"] == "attention_rag.pdf" for r in results)

    @patch("rag.retriever.embed_query")
    def test_paper_filter_works(self, mock_embed, populated_store):
        store, _ = populated_store
        mock_embed.return_value = RNG.random((1, DIM)).astype(np.float32)
        results = retrieve("anything", store, top_k=10,
                           paper_filter="attention_rag.pdf")
        assert all(r["filename"] == "attention_rag.pdf" for r in results)

    @patch("rag.retriever.embed_query")
    def test_unknown_paper_filter_returns_empty(self, mock_embed, populated_store):
        store, _ = populated_store
        mock_embed.return_value = RNG.random((1, DIM)).astype(np.float32)
        results = retrieve("anything", store, top_k=10,
                           paper_filter="does_not_exist.pdf")
        assert results == []


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Reranking
# ══════════════════════════════════════════════════════════════════════════════

class TestStep5_Reranking:

    @patch("rag.retriever.embed_query")
    def test_reranking_reduces_to_top_n(self, mock_embed, populated_store, capsys):
        store, _ = populated_store
        mock_embed.return_value = RNG.random((1, DIM)).astype(np.float32)
        query = "What are the limitations of AttentionRAG?"
        raw_chunks = retrieve(query, store, top_k=8)

        import rag.reranker as reranker_mod
        original_client = reranker_mod._client
        mock_client = MagicMock()
        mock_client.rerank.side_effect = (
            lambda **kw: _mock_cohere_rerank_response(raw_chunks, kw.get("top_n", 3))
        )
        reranker_mod._client = mock_client

        try:
            reranked = rerank(query, raw_chunks, top_n=3)
        finally:
            reranker_mod._client = original_client

        print("\n" + "═" * 60)
        print("STEP 5 — RERANKING (Cohere cross-encoder)")
        print("═" * 60)
        print(f"  Query       : {query}")
        print(f"  Before rerank: {len(raw_chunks)} chunks")
        print(f"  After rerank : {len(reranked)} chunks")
        for i, r in enumerate(reranked):
            print(f"  [{i+1}] rerank_score={r['rerank_score']}  "
                  f"page={r['page_number']}  "
                  f"{r['text'][:70].replace(chr(10),' ')}…")

        assert len(reranked) == 3
        assert all("rerank_score" in r for r in reranked)
        # Scores should be in descending order
        scores = [r["rerank_score"] for r in reranked]
        assert scores == sorted(scores, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — Answer Generation
# ══════════════════════════════════════════════════════════════════════════════

class TestStep6_Generation:

    @patch("rag.retriever.embed_query")
    def test_generates_grounded_answer(self, mock_embed, populated_store, capsys):
        store, _ = populated_store
        mock_embed.return_value = RNG.random((1, DIM)).astype(np.float32)

        query = "What are the main contributions of the AttentionRAG paper?"

        # Build context chunks via retrieval
        raw_chunks = retrieve(query, store, top_k=6)

        # Mock reranker
        import rag.reranker as reranker_mod
        mock_rerank_client = MagicMock()
        mock_rerank_client.rerank.side_effect = (
            lambda **kw: _mock_cohere_rerank_response(raw_chunks, kw.get("top_n", 5))
        )
        original_rerank = reranker_mod._client
        reranker_mod._client = mock_rerank_client

        reranked = rerank(query, raw_chunks, top_n=5)
        reranker_mod._client = original_rerank

        # Mock Groq
        expected_answer = (
            "The AttentionRAG paper makes three key contributions "
            "[Source: attention_rag.pdf, Page 2]: "
            "(1) AttentionRAG — a retrieval-augmented transformer fine-tuned for "
            "scientific text; "
            "(2) SciChunk — an adaptive chunking strategy that preserves section "
            "boundaries [Source: attention_rag.pdf, Page 3]; "
            "(3) SciQA-2024 — a new benchmark dataset with 12,000 expert-annotated "
            "QA pairs [Source: attention_rag.pdf, Page 4]."
        )

        import rag.generator as gen_mod
        original_client = gen_mod._client
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = _mock_groq_response(
            expected_answer
        )
        gen_mod._client = mock_groq

        try:
            answer, used_chunks = generate_answer(query, reranked)
        finally:
            gen_mod._client = original_client

        print("\n" + "═" * 60)
        print("STEP 6 — ANSWER GENERATION (Groq LLM)")
        print("═" * 60)
        print(f"  Query  : {query}")
        print(f"  Answer :\n")
        for line in textwrap.wrap(answer, width=70):
            print(f"    {line}")

        assert isinstance(answer, str) and len(answer) > 20
        assert used_chunks == reranked


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — Citations
# ══════════════════════════════════════════════════════════════════════════════

class TestStep7_Citations:

    def _make_chunks_with_rerank(self) -> list:
        return [
            {"chunk_id": 0, "text": "AttentionRAG contributions…",
             "filename": "attention_rag.pdf", "page_number": 2,
             "rerank_score": 0.95},
            {"chunk_id": 1, "text": "SciChunk methodology…",
             "filename": "attention_rag.pdf", "page_number": 3,
             "rerank_score": 0.87},
            {"chunk_id": 2, "text": "Experimental results…",
             "filename": "attention_rag.pdf", "page_number": 4,
             "rerank_score": 0.73},
            {"chunk_id": 3, "text": "Limitations discussion…",
             "filename": "attention_rag.pdf", "page_number": 5,
             "rerank_score": 0.25},  # < 0.4 → "Low"
        ]

    def test_format_citations_output(self, capsys):
        chunks = self._make_chunks_with_rerank()
        result = format_citations(chunks)

        print("\n" + "═" * 60)
        print("STEP 7 — CITATION FORMATTING")
        print("═" * 60)
        print("\n  --- format_citations() ---")
        print(result)

        assert "attention_rag.pdf" in result
        assert "Page 2" in result
        assert "High" in result      # score 0.95 → High
        assert "Low" in result       # score 0.41 → Low

    def test_source_snippets_output(self, capsys):
        chunks = self._make_chunks_with_rerank()
        result = format_source_snippets(chunks, max_chars=80)

        print("\n  --- format_source_snippets() ---")
        print(result)

        assert "attention_rag.pdf" in result
        assert "Page 3" in result

    def test_apa_citation_format(self, capsys):
        apa = build_apa_citation("attention_rag.pdf", 3)
        print(f"\n  APA  : {apa}")
        assert "Page 3" in apa
        assert "Attention Rag" in apa

    def test_deduplication_works(self):
        chunks = self._make_chunks_with_rerank()
        # Add duplicate page
        chunks.append(dict(chunks[0]))
        result = format_citations(chunks)
        # Page 2 should appear only once
        assert result.count("Page 2") == 1


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — Paper Comparison (single paper, self-compare as proxy)
# ══════════════════════════════════════════════════════════════════════════════

class TestStep8_Comparison:

    @patch("rag.retriever.embed_query")
    def test_compare_two_papers(self, mock_embed, populated_store, tmp_path, capsys):
        """
        Create a second paper and compare them side-by-side.
        """
        store, _ = populated_store
        mock_embed.return_value = RNG.random((1, DIM)).astype(np.float32)

        # Build a second dummy paper and add it to the store
        pdf2_path = str(tmp_path / "baseline_rag.pdf")
        doc = fitz.open()
        page2_texts = [
            textwrap.dedent("""\
                Baseline Dense Retrieval for Scientific QA

                Abstract: We present a simple BM25 + GPT-4 baseline for scientific
                question answering. Our system retrieves relevant passages using
                BM25 and generates answers with GPT-4. We achieve 71.1% exact-match
                accuracy on SciQA-2024, establishing a strong baseline.
            """),
            textwrap.dedent("""\
                Methodology: BM25 Retrieval

                Documents are indexed using BM25 (Robertson et al., 1994).
                Queries are tokenized and matched against the inverted index.
                Top-10 passages are concatenated and passed to GPT-4.
                No fine-tuning or reranking is performed.

                Limitations: BM25 cannot capture semantic similarity. Paraphrased
                queries frequently fail to retrieve relevant passages. The model
                hallucinates when retrieved context is insufficient.
            """),
        ]
        for txt in page2_texts:
            p = doc.new_page(width=595, height=842)
            p.insert_textbox(fitz.Rect(50, 60, 545, 800), txt, fontsize=10)
        doc.save(pdf2_path)
        doc.close()

        # Load + chunk + embed (mocked) the second paper
        from rag.loader import load_pdf as _load
        from rag.chunking import chunk_pages as _chunk

        pages2 = _load(pdf2_path)
        chunks2 = _chunk(pages2, chunk_size=400, chunk_overlap=60)

        import rag.embeddings as emb_mod
        original_client = emb_mod._client
        mock_emb_client = MagicMock()
        mock_emb_client.embed.side_effect = (
            lambda **kw: _mock_cohere_embed_response(kw["texts"])
        )
        emb_mod._client = mock_emb_client

        from rag.embeddings import embed_documents
        embs2 = embed_documents([c["text"] for c in chunks2])
        store.add(embs2, chunks2)
        emb_mod._client = original_client

        # Mock reranker
        import rag.reranker as reranker_mod
        original_rerank = reranker_mod._client
        mock_rerank_client = MagicMock()

        def _dynamic_rerank(**kw):
            docs = kw.get("documents", [])
            n = kw.get("top_n", 3)
            r = MagicMock()
            r.results = [
                _make_result(i, round(0.90 - i * 0.10, 2))
                for i in range(min(n, len(docs)))
            ]
            return r

        def _make_result(idx, score):
            r = MagicMock()
            r.index = idx
            r.relevance_score = score
            return r

        mock_rerank_client.rerank.side_effect = _dynamic_rerank
        reranker_mod._client = mock_rerank_client

        # Mock Groq for comparison
        comparison_answer = (
            "## Methodology\n"
            "Paper A (AttentionRAG) uses adaptive SciChunk + cross-encoder reranking "
            "[Paper A: attention_rag.pdf, Page 3].\n"
            "Paper B (Baseline) uses BM25 retrieval without reranking "
            "[Paper B: baseline_rag.pdf, Page 2].\n\n"
            "## Results & Findings\n"
            "Paper A achieves 78.4% vs Paper B's 71.1% on SciQA-2024 "
            "[Paper A: attention_rag.pdf, Page 4].\n\n"
            "## Limitations\n"
            "Paper A: latency, English-only, no multi-modal support "
            "[Paper A: attention_rag.pdf, Page 5].\n"
            "Paper B: BM25 cannot capture semantics, frequent hallucination "
            "[Paper B: baseline_rag.pdf, Page 2].\n\n"
            "## Key Differences\n"
            "AttentionRAG's reranking and fine-tuned embeddings provide a 7.3% "
            "accuracy advantage over the BM25 baseline."
        )

        import rag.generator as gen_mod
        original_gen = gen_mod._client
        mock_groq = MagicMock()
        mock_groq.chat.completions.create.return_value = _mock_groq_response(
            comparison_answer
        )
        gen_mod._client = mock_groq

        try:
            answer, chunks_a, chunks_b = compare_papers(
                query="How do the methodologies and results compare?",
                paper_a="attention_rag.pdf",
                paper_b="baseline_rag.pdf",
                store=store,
                top_k=6,
                top_n=3,
            )
        finally:
            reranker_mod._client = original_rerank
            gen_mod._client = original_gen

        print("\n" + "═" * 60)
        print("STEP 8 — PAPER COMPARISON")
        print("═" * 60)
        print(f"  Paper A: attention_rag.pdf  ({len(chunks_a)} source chunks)")
        print(f"  Paper B: baseline_rag.pdf   ({len(chunks_b)} source chunks)")
        print("\n  Comparison answer:\n")
        for line in answer.split("\n"):
            print(f"    {line}")

        assert "Methodology" in answer
        assert "Key Differences" in answer
        assert len(chunks_a) > 0
        assert len(chunks_b) > 0


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — Full pipeline wired together (smoke test)
# ══════════════════════════════════════════════════════════════════════════════

class TestStep9_FullPipelineSmokeTest:

    @patch("rag.retriever.embed_query")
    def test_complete_question_answer_flow(self, mock_embed, populated_store, capsys):
        """
        Single test that exercises the entire RAG pipeline in sequence.
        Simulates exactly what the Streamlit UI does on every user question.
        """
        store, _ = populated_store
        mock_embed.return_value = RNG.random((1, DIM)).astype(np.float32)

        question = "What benchmark results does AttentionRAG achieve and how does it compare to GPT-4?"

        # ── 1. Retrieve ────────────────────────────────────────────────────────
        raw = retrieve(question, store, top_k=8)
        assert len(raw) > 0

        # ── 2. Rerank ──────────────────────────────────────────────────────────
        import rag.reranker as reranker_mod
        original_rr = reranker_mod._client
        mc = MagicMock()
        mc.rerank.side_effect = (
            lambda **kw: _mock_cohere_rerank_response(raw, kw.get("top_n", 5))
        )
        reranker_mod._client = mc
        reranked = rerank(question, raw, top_n=5)
        reranker_mod._client = original_rr

        # ── 3. Generate ────────────────────────────────────────────────────────
        expected = (
            "AttentionRAG achieves 78.4% exact-match on SciQA-2024, surpassing "
            "GPT-4 by 7.3% [Source: attention_rag.pdf, Page 4]. "
            "On PubMedQA it scores 82.1% vs GPT-4's 79.5% [Source: attention_rag.pdf, Page 4]."
        )
        import rag.generator as gen_mod
        original_gen = gen_mod._client
        mg = MagicMock()
        mg.chat.completions.create.return_value = _mock_groq_response(expected)
        gen_mod._client = mg
        answer, used = generate_answer(question, reranked)
        gen_mod._client = original_gen

        # ── 4. Citations ───────────────────────────────────────────────────────
        citations = format_citations(used)
        snippets  = format_source_snippets(used, max_chars=150)

        print("\n" + "═" * 60)
        print("STEP 9 — FULL PIPELINE SMOKE TEST")
        print("═" * 60)
        print(f"\n  Q: {question}\n")
        print(f"  A: {answer}\n")
        print(f"  Retrieved : {len(raw)} chunks")
        print(f"  Reranked  : {len(reranked)} chunks")
        print(f"\n  {citations}")

        # ── Assertions ─────────────────────────────────────────────────────────
        assert isinstance(answer, str) and len(answer) > 10
        assert "attention_rag.pdf" in citations
        assert len(snippets) > 0
        print("\n  ✅ Full pipeline executed successfully.")
