# Research Paper RAG Chatbot 📚

A production-quality, lightweight Retrieval-Augmented Generation (RAG) chatbot that lets you upload research PDFs, ask questions, get grounded answers with citations, compare papers, and export notes.

**v2 — Trust & Quality Upgrade:** Grounding guardrail · Confidence labels · Source chunk preview · Streaming · Evaluation dashboard · Rerank comparison · Bibliography browser · Sample demo mode.

---

## Tech Stack

| Layer | Library |
|---|---|
| UI | [Streamlit](https://streamlit.io) |
| LLM generation | [Groq](https://groq.com) (`llama-3.3-70b-versatile`) |
| Embeddings | [Cohere Embed](https://docs.cohere.com/docs/embeddings) (`embed-english-v3.0`) |
| Reranking | [Cohere Rerank](https://docs.cohere.com/docs/rerank) (`rerank-english-v3.0`) |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) (IndexFlatL2) |
| PDF parsing | [PyMuPDF](https://pymupdf.readthedocs.io) |
| Environment | `python-dotenv` |
| Testing | `pytest` + `pytest-mock` |

---

## Project Structure

```
research-paper-rag-chatbot/
├── app.py                  ← Streamlit UI (entry point)
├── requirements.txt
├── .env.example            ← Copy to .env and fill in keys
├── .gitignore
├── README.md
│
├── rag/                    ← Core RAG pipeline
│   ├── __init__.py
│   ├── loader.py           ← PDF text extraction (PyMuPDF)
│   ├── chunking.py         ← Sliding-window text chunking
│   ├── embeddings.py       ← Cohere Embed (documents + queries)
│   ├── vectorstore.py      ← FAISS index wrapper
│   ├── retriever.py        ← Query → FAISS search
│   ├── reranker.py         ← Cohere Rerank
│   ├── generator.py        ← Groq LLM answer generation + streaming
│   ├── citations.py        ← Citation formatting (APA / IEEE)
│   ├── compare.py          ← Side-by-side paper comparison
│   ├── grounding.py        ← ★ Grounding guardrail + confidence labels
│   ├── evaluation.py       ← ★ RAG evaluation metrics
│   ├── references.py       ← ★ Bibliography / reference extraction
│   └── demo.py             ← ★ Sample paper + demo mode
│
├── utils/                  ← Shared helpers
│   ├── __init__.py
│   ├── config.py           ← All settings from .env (incl. trust thresholds)
│   ├── logger.py           ← Structured stdout logging
│   ├── file_utils.py       ← Safe file save / hash / delete
│   └── text_utils.py       ← Text cleaning utilities
│
├── data/
│   ├── uploads/            ← Saved PDFs (git-ignored)
│   ├── index/              ← Persisted FAISS index (git-ignored)
│   ├── metadata/           ← Reserved for future SQLite storage
│   └── sample/             ← Built-in demo PDF (auto-generated)
│
└── tests/
    ├── conftest.py
    ├── test_loader.py
    ├── test_chunking.py
    ├── test_retriever.py
    ├── test_chat_flow.py
    ├── test_e2e_pipeline.py
    ├── test_grounding.py    ← ★ Grounding guardrail tests
    ├── test_evaluation.py   ← ★ Evaluation metrics tests
    ├── test_references.py   ← ★ Bibliography extraction tests
    ├── test_demo.py         ← ★ Sample demo tests
    ├── test_streaming.py    ← ★ Streaming generator tests
    └── test_trust_e2e.py    ← ★ End-to-end trust feature smoke test
```

---

## Quick Start

### 1 — Clone & enter the project

```bash
cd "research-paper-rag-chatbot"
```

### 2 — Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4 — Configure API keys

```bash
cp .env.example .env
```

Open `.env` and fill in:

```env
GROQ_API_KEY=gsk_...          # https://console.groq.com
COHERE_API_KEY=...            # https://dashboard.cohere.com
```

### 5 — Run the app

```bash
streamlit run app.py
```

Open `http://localhost:8501`. Click **Load Sample Paper** to try without uploading files.

---

## How to Use

1. **Demo mode** — click **Load Sample Paper** in the sidebar to load a built-in RAG overview paper instantly.
2. **Upload PDFs** — drag and drop one or more research papers in the sidebar.
3. **Ask questions** — answers include a confidence badge, citations, and source chunk previews.
4. **Check grounding** — if evidence is weak, the bot issues a soft refusal instead of guessing.
5. **Inspect chunks** — expand *Source Chunks* to see exact retrieved text, page numbers, and rerank scores.
6. **Compare reranking** — expand *Rerank Comparison* to see how Cohere Rerank reorders FAISS results.
7. **Use shortcuts** — Summarize / Contributions / Limitations / Methodology buttons for instant queries.
8. **Compare papers** — switch to *Compare Papers* mode for structured side-by-side analysis.
9. **Evaluation dashboard** — after asking questions, expand *RAG Evaluation Dashboard* to see hit rate, precision, groundedness, and fallback rate.
10. **Browse references** — if the paper has a bibliography, it appears in *Extracted References*.
11. **Export** — download the full session as Markdown or JSON (includes confidence scores and references).
12. **Persist the index** — *Save / Load Index* in the sidebar.

---

## New Features (v2)

### Grounding Guardrail
Every answer is scored on an **evidence score** (0–1) derived from:
- Mean and max Cohere rerank score of the retrieved chunks.
- Fraction of chunks that clear a minimum support threshold.

| Evidence Score | Confidence Label | Behaviour |
|---|---|---|
| ≥ 0.55 | 🟢 High | Answer generated normally |
| ≥ 0.20 | 🟡 Medium | Answer generated with caution note |
| ≥ 0.05 | 🔴 Low | Answer generated with warning |
| < 0.05 | 🔴 Low | **Soft refusal** — no LLM call made |

Thresholds are configurable in `.env`:
```env
CONFIDENCE_HIGH_THRESHOLD=0.6
CONFIDENCE_MEDIUM_THRESHOLD=0.35
GROUNDING_REFUSAL_THRESHOLD=0.20
MIN_SUPPORTING_SCORE=0.30
```

### Source Chunk Preview
Each answer expands a **Source Chunks** panel showing:
- Exact chunk text from the paper.
- Filename and page number.
- Cohere rerank score and FAISS L2 distance.
- Per-chunk High / Medium / Low confidence colour.

### Rerank Comparison Panel
The **Rerank Comparison** expander shows FAISS retrieval order vs Cohere rerank order side-by-side, with scores, so you can see exactly how the reranker improved precision.

### Streaming Responses
Groq responses are streamed token-by-token into the Streamlit UI. Toggle **Stream responses** in the sidebar. Automatically falls back to batch if streaming is unavailable.

### RAG Evaluation Dashboard
After each query, metrics are recorded and aggregated:

| Metric | Definition |
|---|---|
| Hit Rate @ K | Fraction of top-N chunks containing query keywords |
| Context Precision | Fraction of chunks ≥ minimum rerank threshold |
| Groundedness | Fraction of chunks that clearly support the answer |
| Fallback Rate | Fraction of queries that triggered soft refusal |
| Avg Rerank Score | Mean Cohere rerank score across queries |
| Rerank Improvement | Score delta between reranked and pre-rerank chunks |

### Reference Extraction
When a PDF contains a **References** or **Bibliography** section, entries are automatically extracted and displayed in the *Extracted References* panel. References are also included in JSON export.

### Sample Demo Mode
Click **Load Sample Paper** to load a built-in 6-page paper on RAG systems, generated at runtime. No upload needed. Sample questions are shown on the welcome screen.

---

## How the RAG Flow Works

```
User uploads PDF  (or clicks Load Sample Paper)
      │
      ▼
PyMuPDF extracts text page-by-page
      │
      ▼
Text cleaned → split into 512-char overlapping chunks
References extracted from bibliography section
      │
      ▼
Cohere embed-english-v3.0 encodes each chunk
(input_type="search_document")
      │
      ▼
Chunk vectors stored in FAISS IndexFlatL2
      │
User types a question
      │
      ▼
Cohere encodes the query (input_type="search_query")
      │
      ▼
FAISS k-NN search → top-10 candidate chunks
      │
      ▼
Cohere Rerank (cross-encoder) → top-5 chunks
      │
      ▼
Evidence score computed → confidence label assigned
If score < refusal threshold → soft refusal returned
      │
      ▼
Groq llama-3.3-70b-versatile generates grounded answer
(streaming by default)
      │
      ▼
Answer annotated with confidence badge
Citations, source chunk preview, rerank comparison shown
Evaluation metrics recorded
```

---

## Running Tests

```bash
pytest -v
```

**180 tests** — all mock external APIs (no real keys needed).

```bash
# Run specific test files
pytest tests/test_grounding.py -v
pytest tests/test_evaluation.py -v
pytest tests/test_references.py -v
pytest tests/test_demo.py -v
pytest tests/test_streaming.py -v
pytest tests/test_trust_e2e.py -v
```

### Live integration test (real APIs)

```bash
python tests/test_live_integration.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | Required. Get from console.groq.com |
| `COHERE_API_KEY` | — | Required. Get from dashboard.cohere.com |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name |
| `GROQ_MAX_TOKENS` | `1024` | Max generation tokens |
| `GROQ_TEMPERATURE` | `0.1` | Sampling temperature |
| `COHERE_EMBED_MODEL` | `embed-english-v3.0` | Cohere embed model |
| `COHERE_RERANK_MODEL` | `rerank-english-v3.0` | Cohere rerank model |
| `CHUNK_SIZE` | `512` | Chunk size in characters |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_RETRIEVE` | `10` | Chunks fetched from FAISS |
| `TOP_K_RERANK` | `5` | Chunks kept after reranking |
| `CONFIDENCE_HIGH_THRESHOLD` | `0.55` | ★ Evidence score for High label |
| `CONFIDENCE_MEDIUM_THRESHOLD` | `0.20` | ★ Evidence score for Medium label |
| `GROUNDING_REFUSAL_THRESHOLD` | `0.05` | ★ Below this → soft refusal |
| `MIN_SUPPORTING_SCORE` | `0.20` | ★ Minimum rerank score to count as support |
| `MIN_QUERY_WORDS` | `2` | ★ Queries shorter than this get a rephrase hint |
│   ├── citations.py        ← Citation formatting (APA / IEEE)
│   └── compare.py          ← Side-by-side paper comparison
│
├── utils/                  ← Shared helpers
│   ├── __init__.py
│   ├── config.py           ← All settings loaded from .env
│   ├── logger.py           ← Structured stdout logging
│   ├── file_utils.py       ← Safe file save / hash / delete
│   └── text_utils.py       ← Text cleaning utilities
│
├── data/
│   ├── uploads/            ← Saved PDFs (git-ignored)
│   ├── index/              ← Persisted FAISS index (git-ignored)
│   └── metadata/           ← Reserved for future SQLite storage
│
└── tests/
    ├── conftest.py          ← sys.path setup
    ├── __init__.py
    ├── test_loader.py       ← PDF extraction tests
    ├── test_chunking.py     ← Chunking behaviour tests
    ├── test_retriever.py    ← FAISS + retrieval tests
    └── test_chat_flow.py    ← End-to-end pipeline tests (mocked APIs)
```

---

## Quick Start

### 1 — Clone & enter the project

```bash
cd "research-paper-rag-chatbot"
```

### 2 — Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3 — Upgrade pip and install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4 — Configure API keys

```bash
cp .env.example .env
```

Open `.env` in any editor and fill in:

```env
GROQ_API_KEY=gsk_...          # https://console.groq.com
COHERE_API_KEY=...            # https://dashboard.cohere.com
```

All other values have sensible defaults and can be left as-is.

### 5 — Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## How to Use

1. **Upload PDFs** — drag and drop one or more research papers in the sidebar.
2. **Ask questions** — type in the chat box; the assistant retrieves relevant passages and answers grounded in the papers.
3. **See citations** — every answer includes the source filename and page number.
4. **Use shortcuts** — click "Summarize", "Contributions", "Limitations", or "Methodology" for instant one-click queries.
5. **Compare papers** — switch to *Compare Papers* mode in the sidebar, pick two papers, and run a structured side-by-side analysis.
6. **Export** — download the full chat as Markdown or JSON at the bottom of the page.
7. **Persist the index** — click *Save Index* in the sidebar to write the FAISS index to `data/index/`. Click *Load Index* on the next session to restore it without re-embedding.

---

## How the RAG Flow Works

```
User uploads PDF
      │
      ▼
PyMuPDF extracts text page-by-page
      │
      ▼
Text cleaned → split into 512-char overlapping chunks
      │
      ▼
Cohere embed-english-v3.0 encodes each chunk
(input_type="search_document")
      │
      ▼
Chunk vectors stored in FAISS IndexFlatL2
      │
User types a question
      │
      ▼
Cohere encodes the query
(input_type="search_query")
      │
      ▼
FAISS k-NN search → top-10 candidate chunks
      │
      ▼
Cohere Rerank (cross-encoder) → top-5 chunks
      │
      ▼
Groq llama-3.3-70b-versatile generates grounded answer
with [Source: filename, Page N] citations
      │
      ▼
Streamlit displays answer + citations + source snippets
```

Key design decisions:
- **Separate embed input types** — `search_document` for indexing, `search_query` for lookup. This follows Cohere's recommended practice and measurably improves recall.
- **Reranking after FAISS** — FAISS retrieves by vector distance; Cohere Rerank applies a cross-encoder that reads the full (query, passage) pair together, fixing ranking mistakes.
- **No hallucination** — The system prompt explicitly instructs the LLM to refuse answering if the context does not support the claim.
- **Per-page metadata** — every chunk stores its filename and page number, enabling precise citations.

---

## Running Tests

```bash
pytest -v
```

All tests mock external APIs (Cohere, Groq) so **no real API keys are required** to run the test suite.

To run a specific test file:

```bash
pytest tests/test_loader.py -v
pytest tests/test_chunking.py -v
pytest tests/test_retriever.py -v
pytest tests/test_chat_flow.py -v
```

### Manual smoke test

1. Start the app: `streamlit run app.py`
2. Upload any research PDF (e.g., a paper from arXiv).
3. Ask: *"What is the main contribution of this paper?"*
4. Verify the answer references the correct filename and page numbers.
5. Upload a second PDF and switch to Compare mode.
6. Run a comparison on "Methodology".
7. Click "Save Index" → stop the app → `Load Index` on restart → confirm chunks are restored.

---

## Environment Variables Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | — | **Required.** Groq API key. |
| `COHERE_API_KEY` | — | **Required.** Cohere API key. |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Groq model name. |
| `GROQ_MAX_TOKENS` | `1024` | Max tokens in LLM response. |
| `GROQ_TEMPERATURE` | `0.1` | LLM sampling temperature. |
| `COHERE_EMBED_MODEL` | `embed-english-v3.0` | Cohere embedding model. |
| `COHERE_RERANK_MODEL` | `rerank-english-v3.0` | Cohere reranking model. |
| `CHUNK_SIZE` | `512` | Target characters per chunk. |
| `CHUNK_OVERLAP` | `64` | Overlap characters between chunks. |
| `TOP_K_RETRIEVE` | `10` | Chunks fetched from FAISS. |
| `TOP_K_RERANK` | `5` | Chunks kept after Cohere Rerank. |
| `MAX_FILE_SIZE_MB` | `50` | Max upload size (informational). |

---

## Troubleshooting

**`GROQ_API_KEY is not set` error**
→ Make sure `.env` exists (not just `.env.example`) and the app was restarted after editing it.

**`No extractable text found` warning for a PDF**
→ The PDF is likely scanned (image-only). Use a tool like Adobe Acrobat or `ocrmypdf` to add a text layer before uploading.

**`faiss-cpu` import error on Apple Silicon**
→ Install via: `pip install faiss-cpu --no-binary :all:`

**Cohere 429 rate-limit during embedding**
→ Reduce batch size by setting `CHUNK_SIZE=256` or upload one paper at a time.

**Groq context length exceeded**
→ Reduce `TOP_K_RERANK` to 3 or lower `GROQ_MAX_TOKENS`.

**Adding more papers to an existing session**
→ Simply upload additional PDFs in the sidebar — they are appended to the same FAISS index automatically.

---

## Extending the Project

- **Swap the LLM** — replace `groq` with any OpenAI-compatible client in `rag/generator.py`.
- **Add multilingual support** — change `COHERE_EMBED_MODEL` to `embed-multilingual-v3.0`.
- **Scale the index** — replace `IndexFlatL2` with `IndexIVFFlat` in `rag/vectorstore.py` for >100 k vectors.
- **Add a database** — extend `utils/` with a SQLite module to persist chat history across sessions.
- **OCR support** — pipe scanned PDFs through `ocrmypdf` before calling `load_pdf()`.
