"""
Live integration test — calls real Cohere and Groq APIs.

Creates a dummy research paper PDF, runs the full pipeline with NO mocks,
and prints every intermediate result so you can verify the real API responses.

Run with:
    python tests/test_live_integration.py
"""

import os
import sys
import textwrap
import time

# ── Make project root importable ──────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz  # PyMuPDF
from dotenv import load_dotenv

load_dotenv()

from rag.loader import load_pdf
from rag.chunking import chunk_pages
from rag.embeddings import embed_documents, embed_query
from rag.vectorstore import FAISSVectorStore
from rag.retriever import retrieve
from rag.reranker import rerank
from rag.generator import generate_answer
from rag.citations import format_citations, format_source_snippets, build_apa_citation
from rag.compare import compare_papers
from utils.config import GROQ_API_KEY, COHERE_API_KEY

# ─── Helpers ──────────────────────────────────────────────────────────────────

def banner(title: str) -> None:
    print(f"\n{'═' * 65}")
    print(f"  {title}")
    print(f"{'═' * 65}")

def ok(msg: str) -> None:
    print(f"  ✅  {msg}")

def info(msg: str) -> None:
    print(f"  ℹ️   {msg}")

def section(msg: str) -> None:
    print(f"\n  ── {msg}")

# ─── Dummy paper content ───────────────────────────────────────────────────────

_PAPER_A_PAGES = [
    textwrap.dedent("""\
        Attention-Augmented Transformers for Scientific Document Understanding

        Authors: A. Researcher, B. Scientist
        Institution: Institute of AI Research, 2024

        Abstract

        We present AttentionRAG, a retrieval-augmented transformer for scientific
        document understanding. Our model achieves state-of-the-art results on
        SciQA-2024, surpassing GPT-4 by 7.3% exact-match accuracy. The key
        innovation is adaptive chunking (SciChunk) combined with a fine-tuned
        cross-encoder reranker trained on scientific text corpora.
    """),
    textwrap.dedent("""\
        1. Introduction

        Scientific document understanding requires integrating knowledge across
        multiple papers, handling domain-specific terminology, and reasoning over
        complex claims. Prior RAG systems fail on scientific text for three reasons:

        (1) Fixed-size chunking ignores section boundaries, breaking coherent units.
        (2) General-purpose embeddings miss domain-specific semantics.
        (3) Rerankers trained on web text do not generalise to scientific passages.

        Our contributions:
        - AttentionRAG: a retrieval-augmented transformer fine-tuned for science.
        - SciChunk: adaptive chunking that preserves section-level coherence.
        - SciQA-2024: a benchmark with 12,000 expert-annotated question-answer pairs.
    """),
    textwrap.dedent("""\
        2. Methodology

        2.1 Adaptive Chunking (SciChunk)
        SciChunk detects section boundaries using a lightweight CNN classifier
        trained on 50,000 scientific papers. Chunks form at section transitions
        with 10% overlap to avoid context loss. Mean chunk size: 412 tokens.

        2.2 Embeddings
        We fine-tune a 110M parameter encoder on S2ORC (81M scientific papers).
        Embeddings are 1024-dimensional and trained with contrastive loss on
        (query, relevant passage) pairs from scientific QA datasets.

        2.3 Reranking
        A cross-encoder reranked on (query, scientific passage) pairs from PubMed
        and arXiv. The cross-encoder jointly encodes query and passage, providing
        precise relevance scores that bi-encoders cannot achieve.
    """),
    textwrap.dedent("""\
        3. Results

        AttentionRAG vs baselines (exact-match accuracy):
        - SciQA-2024:  AttentionRAG 78.4%  vs GPT-4 71.1%  (+7.3%)
        - PubMedQA:    AttentionRAG 82.1%  vs GPT-4 79.5%  (+2.6%)
        - QASPER:      AttentionRAG 54.3%  vs GPT-4 51.8%  (+2.5%)

        Ablation study:
        - SciChunk alone:    +3.1% over fixed-size chunking.
        - Scientific embeds: +2.4% over general-purpose embeddings.
        - Cross-encoder:     +1.8% over bi-encoder reranking.
    """),
    textwrap.dedent("""\
        4. Limitations and Conclusion

        Limitations:
        (1) Latency: cross-encoder reranking adds ~200ms per query on A100 GPU.
        (2) Language: model trained on English-only scientific papers.
        (3) Multi-modal: figures, equations, and tables are currently ignored.
        (4) Hallucination: occasionally generates incorrect citations for rare topics.

        Conclusion:
        AttentionRAG achieves consistent gains over strong baselines across three
        scientific QA benchmarks. Future work will extend to multi-modal inputs
        and non-English scientific corpora. Model weights released open-source.

        Acknowledgements: NSF Grant #2024-AI-9812.
    """),
]

_PAPER_B_PAGES = [
    textwrap.dedent("""\
        BM25-GPT: A Simple Baseline for Scientific Question Answering

        Authors: X. Baseline, Y. Simple
        Institution: Basic AI Lab, 2024

        Abstract

        We present a simple yet strong baseline for scientific QA combining BM25
        sparse retrieval with GPT-4 generation. Our system achieves 71.1% exact-
        match on SciQA-2024 without any domain-specific fine-tuning, demonstrating
        the power of off-the-shelf components for scientific document understanding.
    """),
    textwrap.dedent("""\
        Methodology

        Retrieval: Documents are indexed with BM25 (Robertson et al., 1994).
        Queries are tokenised and matched via inverted index. Top-10 passages
        are retrieved per query. No neural components or reranking are used.

        Generation: Retrieved passages are concatenated and passed to GPT-4 as
        context. GPT-4 generates answers using its standard instruction-following
        capability with no fine-tuning.

        Limitations:
        - BM25 cannot capture semantic similarity; paraphrased queries fail.
        - No reranking means irrelevant passages frequently reach the LLM.
        - GPT-4 hallucination rate increases when retrieved context is weak.
        - System cannot handle multi-hop reasoning across multiple papers.
    """),
]


def create_pdf(path: str, pages: list[str]) -> str:
    doc = fitz.open()
    for text in pages:
        page = doc.new_page(width=595, height=842)
        page.insert_textbox(fitz.Rect(50, 60, 545, 800), text, fontsize=10)
    doc.save(path)
    doc.close()
    return path


# ══════════════════════════════════════════════════════════════════════════════
# Main integration test
# ══════════════════════════════════════════════════════════════════════════════

def run() -> None:
    # ── Preflight ─────────────────────────────────────────────────────────────
    banner("PREFLIGHT — API KEY CHECK")
    assert GROQ_API_KEY,   "GROQ_API_KEY not set in .env"
    assert COHERE_API_KEY, "COHERE_API_KEY not set in .env"
    ok(f"GROQ_API_KEY   : {GROQ_API_KEY[:12]}…")
    ok(f"COHERE_API_KEY : {COHERE_API_KEY[:12]}…")

    import tempfile
    tmp = tempfile.mkdtemp()
    pdf_a = os.path.join(tmp, "attention_rag.pdf")
    pdf_b = os.path.join(tmp, "bm25_gpt_baseline.pdf")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1 — Load PDFs
    # ══════════════════════════════════════════════════════════════════════════
    banner("STEP 1 — PDF LOADING")
    create_pdf(pdf_a, _PAPER_A_PAGES)
    create_pdf(pdf_b, _PAPER_B_PAGES)

    pages_a = load_pdf(pdf_a)
    pages_b = load_pdf(pdf_b)

    ok(f"attention_rag.pdf    → {len(pages_a)} pages")
    ok(f"bm25_gpt_baseline.pdf → {len(pages_b)} pages")

    for p in pages_a:
        preview = p["text"][:70].replace("\n", " ")
        info(f"  Page {p['page_number']}: {preview}…")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2 — Chunking
    # ══════════════════════════════════════════════════════════════════════════
    banner("STEP 2 — CHUNKING")
    chunks_a = chunk_pages(pages_a, chunk_size=400, chunk_overlap=60)
    chunks_b = chunk_pages(pages_b, chunk_size=400, chunk_overlap=60)
    all_chunks = chunks_a + chunks_b

    ok(f"attention_rag.pdf    → {len(chunks_a)} chunks")
    ok(f"bm25_gpt_baseline.pdf → {len(chunks_b)} chunks")
    ok(f"Total chunks          : {len(all_chunks)}")

    sizes = [len(c["text"]) for c in all_chunks]
    info(f"Chunk sizes → min:{min(sizes)}  max:{max(sizes)}  avg:{sum(sizes)//len(sizes)} chars")

    section("First 2 chunks from Paper A:")
    for c in chunks_a[:2]:
        print(f"\n    [Chunk {c['chunk_id']} | page {c['page_number']}]")
        print(textwrap.indent(textwrap.fill(c["text"][:200], 60), "    "))

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3 — Real Cohere Embeddings
    # ══════════════════════════════════════════════════════════════════════════
    banner("STEP 3 — COHERE EMBEDDINGS (real API call)")
    info("Embedding all chunks with embed-english-v3.0 …")
    t0 = time.time()
    all_texts = [c["text"] for c in all_chunks]
    embeddings = embed_documents(all_texts)
    elapsed = time.time() - t0

    ok(f"Embeddings shape : {embeddings.shape}  (expected: ({len(all_chunks)}, 1024))")
    ok(f"API call took    : {elapsed:.2f}s")
    ok(f"First vector (first 6 dims) : {embeddings[0][:6].tolist()}")

    assert embeddings.shape == (len(all_chunks), 1024), "Wrong embedding shape!"

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4 — FAISS Indexing
    # ══════════════════════════════════════════════════════════════════════════
    banner("STEP 4 — FAISS INDEXING")
    store = FAISSVectorStore(embedding_dim=1024)
    store.add(embeddings, all_chunks)

    ok(f"Vectors in index : {store.index.ntotal}")
    ok(f"Papers indexed   : {store.paper_names}")

    index_dir = os.path.join(tmp, "faiss_index")
    store.save(index_dir)
    reloaded = FAISSVectorStore.load(index_dir)
    ok(f"Save/load roundtrip: {reloaded.total_chunks} chunks restored ✓")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Real Query Embedding + Retrieval
    # ══════════════════════════════════════════════════════════════════════════
    banner("STEP 5 — RETRIEVAL (real Cohere query embed + FAISS)")
    query = "What methodology does AttentionRAG use for chunking and reranking?"
    info(f"Query: {query}")

    t0 = time.time()
    raw_chunks = retrieve(query, store, top_k=8)
    elapsed = time.time() - t0

    ok(f"Retrieved {len(raw_chunks)} chunks in {elapsed:.2f}s")
    section("Top 5 retrieved chunks:")
    for i, c in enumerate(raw_chunks[:5], 1):
        print(f"\n    [{i}] {c['filename']} | page {c['page_number']} | L2={c['score']:.4f}")
        print(f"    {c['text'][:120].replace(chr(10), ' ')}…")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6 — Real Cohere Reranking
    # ══════════════════════════════════════════════════════════════════════════
    banner("STEP 6 — COHERE RERANK (real API call)")
    info(f"Reranking {len(raw_chunks)} chunks → keeping top 5 …")

    t0 = time.time()
    reranked = rerank(query, raw_chunks, top_n=5)
    elapsed = time.time() - t0

    ok(f"Reranked in {elapsed:.2f}s")
    section("Reranked results (best first):")
    for i, c in enumerate(reranked, 1):
        print(f"\n    [{i}] rerank_score={c['rerank_score']}  "
              f"{c['filename']} page {c['page_number']}")
        print(f"    {c['text'][:120].replace(chr(10), ' ')}…")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 — Real Groq Answer Generation
    # ══════════════════════════════════════════════════════════════════════════
    banner("STEP 7 — GROQ ANSWER GENERATION (real API call)")
    info(f"Calling llama-3.3-70b-versatile …")

    t0 = time.time()
    answer, used_chunks = generate_answer(query, reranked)
    elapsed = time.time() - t0

    ok(f"Answer generated in {elapsed:.2f}s ({len(answer)} chars)")
    print(f"\n  {'─'*60}")
    print(f"  QUESTION:\n    {query}")
    print(f"\n  ANSWER:")
    for line in textwrap.wrap(answer, width=65):
        print(f"    {line}")
    print(f"  {'─'*60}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 8 — Citations
    # ══════════════════════════════════════════════════════════════════════════
    banner("STEP 8 — CITATION FORMATTING")
    citations = format_citations(used_chunks)
    snippets  = format_source_snippets(used_chunks, max_chars=150)

    print(f"\n  {citations}")
    section("APA citations:")
    seen: set = set()
    idx = 1
    for c in used_chunks:
        key = (c["filename"], c["page_number"])
        if key not in seen:
            seen.add(key)
            print(f"    {build_apa_citation(c['filename'], c['page_number'])}")
            idx += 1

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 9 — Two more questions
    # ══════════════════════════════════════════════════════════════════════════
    banner("STEP 9 — ADDITIONAL Q&A (2 more questions)")

    extra_questions = [
        "What are the benchmark results of AttentionRAG compared to GPT-4?",
        "What are the limitations of both papers?",
    ]

    for q in extra_questions:
        info(f"Q: {q}")
        raw    = retrieve(q, store, top_k=6)
        ranked = rerank(q, raw, top_n=3)
        ans, _ = generate_answer(q, ranked)
        print(f"\n  A: ")
        for line in textwrap.wrap(ans, 65):
            print(f"    {line}")
        print()
        time.sleep(1)  # gentle rate limiting

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 10 — Real Paper Comparison
    # ══════════════════════════════════════════════════════════════════════════
    banner("STEP 10 — PAPER COMPARISON (real API calls)")
    compare_q = "How do the methodologies and results of the two papers differ?"
    info(f"Comparing attention_rag.pdf vs bm25_gpt_baseline.pdf")
    info(f"Question: {compare_q}")

    t0 = time.time()
    comp_answer, ca, cb = compare_papers(
        query=compare_q,
        paper_a="attention_rag.pdf",
        paper_b="bm25_gpt_baseline.pdf",
        store=store,
        top_k=6,
        top_n=3,
    )
    elapsed = time.time() - t0

    ok(f"Comparison done in {elapsed:.2f}s")
    ok(f"Paper A sources: {len(ca)} chunks | Paper B sources: {len(cb)} chunks")
    print(f"\n  {'─'*60}")
    for line in comp_answer.split("\n"):
        print(f"    {line}")
    print(f"  {'─'*60}")

    # ══════════════════════════════════════════════════════════════════════════
    # Final summary
    # ══════════════════════════════════════════════════════════════════════════
    banner("ALL STEPS COMPLETE")
    ok("PDF Loading          — real PyMuPDF extraction")
    ok("Chunking             — sliding-window with boundary detection")
    ok("Cohere Embed         — real API, 1024-dim vectors")
    ok("FAISS Indexing       — vectors stored and retrieved")
    ok("FAISS Retrieval      — nearest-neighbour search")
    ok("Cohere Rerank        — real cross-encoder API")
    ok("Groq Generation      — real LLM answer with citations")
    ok("Citation Formatting  — APA + confidence labels")
    ok("Paper Comparison     — dual-retrieval + structured output")
    print(f"\n  🎉  Full end-to-end pipeline working with real APIs!\n")


if __name__ == "__main__":
    run()
