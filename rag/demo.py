"""
Sample / demo mode utilities.

Provides a pre-built sample PDF (generated at runtime with PyMuPDF)
so first-time users can try the chatbot without uploading a paper.

The sample paper is a synthetic 6-page document about "Retrieval-Augmented
Generation for Knowledge-Intensive NLP Tasks" — covering the key ideas of
RAG systems in enough depth to demonstrate all chatbot features.
"""

import io
import os
from typing import List, Dict, Any

from utils.logger import get_logger

logger = get_logger(__name__)

SAMPLE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sample",
)
SAMPLE_FILENAME = "RAG_Overview_Sample.pdf"
SAMPLE_PATH = os.path.join(SAMPLE_DIR, SAMPLE_FILENAME)

# ─── Sample paper text (6 pages) ──────────────────────────────────────────────

_PAGES: List[Dict[str, str]] = [
    {
        "title": "Abstract & Introduction",
        "body": (
            "Retrieval-Augmented Generation (RAG) is a hybrid NLP architecture "
            "that combines parametric memory (a pre-trained language model) with "
            "non-parametric memory (a dense retrieval index over a document corpus). "
            "RAG models retrieve relevant documents at inference time and condition "
            "generation on those documents, achieving strong performance on "
            "knowledge-intensive tasks without requiring the model to memorise "
            "all world knowledge during pre-training.\n\n"
            "Traditional closed-book generation models such as GPT-2 and T5 rely "
            "entirely on knowledge encoded in model weights. This approach leads to "
            "hallucinations when the required knowledge was absent or outdated in "
            "the training data. RAG addresses this by grounding generation in "
            "retrieved evidence, dramatically reducing factual errors and enabling "
            "real-time knowledge updates by swapping the document index.\n\n"
            "This paper provides an overview of the RAG paradigm, its core "
            "components, evaluation methodology, and key findings across several "
            "benchmark datasets including Natural Questions, TriviaQA, WebQ, "
            "and CuratedTrec."
        ),
    },
    {
        "title": "Related Work",
        "body": (
            "Dense Passage Retrieval (DPR) introduced bi-encoder retrieval using "
            "dual BERT encoders trained on question-passage pairs. DPR showed that "
            "learned dense representations outperform classical sparse methods "
            "such as BM25 on open-domain QA benchmarks.\n\n"
            "REALM (Retrieval-Enhanced Language Model) pre-trains a language model "
            "jointly with a retriever in an end-to-end masked language modelling "
            "objective, demonstrating that retrieval can be integrated into "
            "pre-training. However, REALM is computationally expensive.\n\n"
            "FiD (Fusion-in-Decoder) extends T5 with multi-document fusion, "
            "encoding each retrieved passage independently and fusing them in the "
            "decoder cross-attention. FiD achieves state-of-the-art results on "
            "NaturalQuestions and TriviaQA.\n\n"
            "Unlike these, RAG integrates retrieval directly into the generation "
            "process using a trainable retriever and a seq2seq generator, providing "
            "end-to-end differentiability."
        ),
    },
    {
        "title": "Architecture & Methodology",
        "body": (
            "The RAG model has two main components:\n\n"
            "1. Retriever: A DPR bi-encoder that maps queries and documents to a "
            "shared embedding space. Given a query q, the retriever returns the top-k "
            "documents z_1, ..., z_k from a pre-built FAISS index over a large "
            "document corpus (e.g., Wikipedia).\n\n"
            "2. Generator: A BART-large seq2seq model that conditions generation on "
            "the concatenation of the query and each retrieved document. Two variants "
            "are proposed:\n"
            "   - RAG-Sequence: marginalises over documents for the entire sequence.\n"
            "   - RAG-Token: marginalises over documents for each generated token.\n\n"
            "The index is built offline using FAISS IndexFlatIP over 100-word "
            "document chunks from the December 2018 Wikipedia dump, resulting in "
            "21 million passages. Retrieval is performed in sub-second latency "
            "with an approximate MIPS index.\n\n"
            "During training, the retriever and generator are jointly optimised "
            "by minimising the negative log-likelihood of the target answer, "
            "marginalised over the top-k retrieved documents. The FAISS index "
            "is periodically refreshed to reflect updated document embeddings."
        ),
    },
    {
        "title": "Experiments & Results",
        "body": (
            "RAG models are evaluated on four open-domain QA benchmarks:\n\n"
            "NaturalQuestions (NQ): RAG-Token achieves 44.5 EM, outperforming "
            "the closed-book T5-11B (36.6 EM) and the DPR reader (41.5 EM).\n\n"
            "TriviaQA: RAG-Token achieves 56.8 EM, close to the state-of-the-art "
            "at time of publication and substantially above the T5-11B baseline.\n\n"
            "WebQ: RAG-Token achieves 45.5 EM, a strong result given that WebQ "
            "contains questions with structured Freebase answers.\n\n"
            "CuratedTrec: RAG-Sequence achieves 68.0 EM, benefiting from the "
            "sequence-level marginalisation which suits fact retrieval tasks.\n\n"
            "On the MSMARCO passage ranking task, RAG-Sequence achieves "
            "state-of-the-art results in the abstractive generation category, "
            "demonstrating that it can generate fluent natural language answers "
            "rather than merely extracting spans.\n\n"
            "Human evaluation shows RAG answers are rated as more factual and "
            "more specific than closed-book T5 answers, confirming that grounding "
            "generation in retrieved evidence leads to higher quality outputs."
        ),
    },
    {
        "title": "Limitations & Discussion",
        "body": (
            "While RAG demonstrates strong empirical results, several limitations "
            "remain:\n\n"
            "Knowledge Staleness: The document index is static. For rapidly "
            "changing domains (e.g., recent events), the index must be rebuilt "
            "frequently, which is computationally expensive at Wikipedia scale.\n\n"
            "Retrieval Failures: When the correct passage is not retrieved in the "
            "top-k, the generator has no evidence to ground its answer, leading to "
            "hallucination. Increasing k improves recall but increases latency.\n\n"
            "Multi-hop Reasoning: RAG struggles on tasks requiring reasoning across "
            "multiple documents simultaneously, as each retrieved passage is encoded "
            "independently by BART.\n\n"
            "Domain Shift: Performance degrades when the query distribution differs "
            "significantly from the Wikipedia-based index. Custom domain corpora "
            "require rebuilding the index and potentially fine-tuning the retriever.\n\n"
            "Interpretability: The marginalisation over top-k documents makes it "
            "difficult to attribute individual generation decisions to specific "
            "retrieved passages.\n\n"
            "Future work includes dynamic index updates, improved multi-hop "
            "reasoning via iterative retrieval, and uncertainty-aware generation "
            "that signals low-confidence answers."
        ),
    },
    {
        "title": "Conclusion & References",
        "body": (
            "We presented Retrieval-Augmented Generation (RAG), a general-purpose "
            "architecture for knowledge-intensive NLP. RAG combines a learned "
            "dense retriever with a seq2seq generator, jointly trained end-to-end. "
            "Experiments show that RAG achieves state-of-the-art results on "
            "multiple open-domain QA benchmarks, outperforming both large "
            "closed-book models and extractive readers.\n\n"
            "The key insight is that language models do not need to memorise "
            "all world knowledge; instead, they can reliably ground generation "
            "in retrieved evidence, leading to more accurate, specific, and "
            "verifiable answers.\n\n"
            "References\n\n"
            "[1] Lewis, P. et al. (2020). Retrieval-Augmented Generation for "
            "Knowledge-Intensive NLP Tasks. NeurIPS 2020.\n"
            "[2] Karpukhin, V. et al. (2020). Dense Passage Retrieval for "
            "Open-Domain Question Answering. EMNLP 2020.\n"
            "[3] Izacard, G. & Grave, E. (2021). Leveraging Passage Retrieval "
            "with Generative Models for Open Domain Question Answering. EACL 2021.\n"
            "[4] Guu, K. et al. (2020). REALM: Retrieval-Augmented Language "
            "Model Pre-Training. ICML 2020.\n"
            "[5] Johnson, J. et al. (2019). Billion-Scale Similarity Search "
            "with GPUs. IEEE TPAMI.\n"
            "[6] Lewis, M. et al. (2020). BART: Denoising Sequence-to-Sequence "
            "Pre-training for Natural Language Generation. ACL 2020."
        ),
    },
]


# ─── PDF generation ────────────────────────────────────────────────────────────

def _build_sample_pdf() -> bytes:
    """
    Generate the sample paper as a PDF in memory using PyMuPDF (fitz).

    Returns:
        Raw bytes of the generated PDF.
    """
    import fitz  # PyMuPDF

    doc = fitz.open()

    for page_data in _PAGES:
        page = doc.new_page(width=595, height=842)  # A4

        # Title
        page.insert_text(
            (50, 60),
            page_data["title"],
            fontsize=14,
            fontname="helv",
            color=(0, 0, 0.6),
        )

        # Paper title at top of first page
        if page_data["title"] == "Abstract & Introduction":
            page.insert_text(
                (50, 40),
                "Retrieval-Augmented Generation: An Overview",
                fontsize=11,
                fontname="helv",
                color=(0.3, 0.3, 0.3),
            )

        # Body — insert lines manually (fitz does not word-wrap)
        y = 90
        for line in page_data["body"].split("\n"):
            # Wrap long lines
            while len(line) > 85:
                page.insert_text((50, y), line[:85], fontsize=10, fontname="helv")
                line = line[85:]
                y += 14
            if line:
                page.insert_text((50, y), line, fontsize=10, fontname="helv")
            y += 14
            if y > 800:
                break  # Prevent overflow

    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


def ensure_sample_pdf() -> str:
    """
    Ensure the sample PDF exists on disk.  Creates it if missing.

    Returns:
        Absolute path to the sample PDF.
    """
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    if not os.path.exists(SAMPLE_PATH):
        logger.info(f"Generating sample PDF at: {SAMPLE_PATH}")
        pdf_bytes = _build_sample_pdf()
        with open(SAMPLE_PATH, "wb") as f:
            f.write(pdf_bytes)
    return SAMPLE_PATH


def get_sample_pdf_bytes() -> bytes:
    """Return the sample PDF as raw bytes (generates if needed)."""
    path = ensure_sample_pdf()
    with open(path, "rb") as f:
        return f.read()


# ─── Sample questions ──────────────────────────────────────────────────────────

SAMPLE_QUESTIONS: List[str] = [
    "What is Retrieval-Augmented Generation and how does it work?",
    "What are the two main components of the RAG architecture?",
    "How does RAG compare to closed-book generation models?",
    "What are the limitations of RAG systems?",
    "What benchmark datasets were used to evaluate RAG?",
]
