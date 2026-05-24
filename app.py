"""
Research Paper RAG Chatbot — Streamlit Application Entry Point.

Run with:
    streamlit run app.py

New features in v2 (trust & quality upgrade)
--------------------------------------------
- Grounding guardrail: soft refusal when evidence is weak.
- Confidence labels (High / Medium / Low) on every answer.
- Source chunk preview panel with scores and page numbers.
- RAG evaluation dashboard (hit rate, precision, groundedness, etc.).
- Streaming Groq responses.
- Rerank comparison panel showing before/after scores.
- Reference / bibliography browser.
- Sample demo mode for first-time users.
"""

import io
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import streamlit as st

from rag.loader import load_pdf
from rag.chunking import chunk_pages
from rag.embeddings import embed_documents
from rag.vectorstore import FAISSVectorStore
from rag.retriever import retrieve
from rag.reranker import rerank
from rag.generator import generate_answer, stream_answer
from rag.citations import (
    format_citations,
    format_source_snippets,
    build_apa_citation,
    build_ieee_citation,
)
from rag.compare import compare_papers
from rag.grounding import evaluate_grounding, build_refusal_message, annotate_answer
from rag.evaluation import compute_query_metrics, aggregate_session_metrics
from rag.references import extract_references, format_references_markdown
from rag.demo import (
    ensure_sample_pdf,
    get_sample_pdf_bytes,
    SAMPLE_FILENAME,
    SAMPLE_QUESTIONS,
)
from utils.config import (
    UPLOAD_DIR,
    INDEX_DIR,
    GROQ_API_KEY,
    COHERE_API_KEY,
    TOP_K_RETRIEVE,
    TOP_K_RERANK,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    APP_TITLE,
)
from utils.file_utils import save_uploaded_file, get_file_hash, ensure_dir
from utils.logger import get_logger

logger = get_logger(__name__)

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_session() -> None:
    defaults: Dict[str, Any] = {
        "store": FAISSVectorStore(),
        "messages": [],
        "papers": {},
        "references": {},
        "processed_hashes": set(),
        "mode": "chat",
        "top_k": TOP_K_RETRIEVE,
        "top_n": TOP_K_RERANK,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "citation_style": "APA",
        "use_streaming": True,
        "eval_history": [],
        "raw_chunks_last": [],
        "reranked_last": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


def _check_api_keys() -> bool:
    """
    Validate that all required API keys are configured.
    Supports both local development (.env) and Streamlit Cloud (secrets).
    """
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not COHERE_API_KEY:
        missing.append("COHERE_API_KEY")
    
    if missing:
        st.error(f"❌ **Missing API key(s): {', '.join(missing)}**")
        st.info(
            "### Setup Instructions\n\n"
            "**For Local Development:**\n"
            "1. Copy `.env.example` to `.env`\n"
            "2. Add your API keys to `.env`:\n"
        )
        st.code(
            "\n".join(f"{k}=your_key_here" for k in missing),
            language="bash"
        )
        st.info(
            "3. Restart Streamlit: `streamlit run app.py`\n\n"
            "**For Streamlit Community Cloud:**\n"
            "1. Go to your app's **Settings** → **Secrets**\n"
            "2. Add each key in TOML format:\n"
        )
        st.code(
            "\n".join(f'{k} = "your_key_here"' for k in missing),
            language="toml"
        )
        st.stop()
    
    return True


def _process_pdf(uploaded_file, file_bytes: Optional[bytes] = None) -> bool:
    if file_bytes is None:
        file_bytes = uploaded_file.read()
    filename: str = (
        uploaded_file.name if hasattr(uploaded_file, "name") else uploaded_file
    )
    file_hash: str = get_file_hash(file_bytes)

    if file_hash in st.session_state.processed_hashes:
        st.info(f"ℹ️ '{filename}' is already indexed — skipping.")
        return True

    if filename in st.session_state.papers:
        st.warning(f"⚠️ A file named '{filename}' is already loaded.")
        return False

    ensure_dir(UPLOAD_DIR)
    file_path = save_uploaded_file(UPLOAD_DIR, file_bytes, filename)

    try:
        with st.spinner(f"📄 Extracting text from **{filename}** …"):
            pages = load_pdf(file_path)

        if not pages:
            st.error(f"❌ No extractable text in **{filename}**.")
            return False

        with st.spinner(f"✂️ Chunking **{filename}** …"):
            chunks = chunk_pages(
                pages,
                chunk_size=st.session_state.chunk_size,
                chunk_overlap=st.session_state.chunk_overlap,
            )

        if not chunks:
            st.error(f"❌ No usable chunks from **{filename}**.")
            return False

        with st.spinner(f"🧠 Embedding **{filename}** ({len(chunks)} chunks) …"):
            texts = [c["text"] for c in chunks]
            embeddings = embed_documents(texts)

        with st.spinner(f"💾 Indexing **{filename}** …"):
            st.session_state.store.add(embeddings, chunks)

        refs = extract_references(pages, filename)
        if refs:
            st.session_state.references[filename] = refs

        st.session_state.papers[filename] = {
            "filename": filename,
            "chunks": len(chunks),
            "pages": pages[-1]["total_pages"] if pages else 0,
            "hash": file_hash,
            "indexed_at": datetime.now().strftime("%H:%M:%S"),
            "has_refs": bool(refs),
            "ref_count": len(refs),
        }
        st.session_state.processed_hashes.add(file_hash)

        ref_note = f" | {len(refs)} refs" if refs else ""
        st.success(
            f"✅ **{filename}** indexed! "
            f"({len(chunks)} chunks, {len(pages)} pages{ref_note})"
        )
        return True

    except Exception as exc:
        st.error(f"❌ Error processing **{filename}**: {exc}")
        logger.exception(f"Failed to process '{filename}'")
        return False


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("📚 Research RAG")
        st.caption("Upload papers, ask questions, compare findings.")
        st.divider()

        st.subheader("🚀 Quick Start")
        if st.button(
            "Load Sample Paper",
            help="Load a built-in RAG overview paper to try instantly.",
            use_container_width=True,
        ):
            if SAMPLE_FILENAME not in st.session_state.papers:
                with st.spinner("Preparing sample paper …"):
                    try:
                        pdf_bytes = get_sample_pdf_bytes()
                        _process_pdf(SAMPLE_FILENAME, file_bytes=pdf_bytes)
                    except Exception as e:
                        st.error(f"Demo load failed: {e}")
            else:
                st.info("Sample paper already loaded.")

        st.divider()
        st.subheader("Upload Papers")
        uploaded_files = st.file_uploader(
            "Drop PDF(s) here",
            type=["pdf"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploaded_files:
            for f in uploaded_files:
                _process_pdf(f)

        if st.session_state.papers:
            st.divider()
            st.subheader(f"Loaded Papers ({len(st.session_state.papers)})")
            for fname, meta in st.session_state.papers.items():
                with st.expander(f"📄 {fname}"):
                    st.caption(
                        f"Pages: {meta['pages']} | Chunks: {meta['chunks']} | "
                        f"Indexed: {meta['indexed_at']}"
                    )
                    if meta.get("has_refs"):
                        st.caption(f"📖 {meta['ref_count']} references extracted")

        st.divider()
        st.subheader("Mode")
        mode_choice = st.radio(
            "Mode",
            options=["💬 Chat", "⚖️ Compare Papers"],
            label_visibility="collapsed",
        )
        st.session_state.mode = "chat" if "Chat" in mode_choice else "compare"

        st.divider()
        st.subheader("Settings")

        with st.expander("Retrieval"):
            st.session_state.top_k = st.slider(
                "Top-K retrieve", 3, 20, value=st.session_state.top_k,
            )
            st.session_state.top_n = st.slider(
                "Top-N rerank", 1, 10, value=st.session_state.top_n,
            )

        with st.expander("Chunking (apply before uploading)"):
            st.session_state.chunk_size = st.slider(
                "Chunk size (chars)", 256, 1024,
                value=st.session_state.chunk_size, step=64,
            )
            st.session_state.chunk_overlap = st.slider(
                "Chunk overlap (chars)", 0, 256,
                value=st.session_state.chunk_overlap, step=16,
            )

        with st.expander("Generation"):
            st.session_state.use_streaming = st.toggle(
                "Stream responses",
                value=st.session_state.use_streaming,
                help="Show tokens as they arrive from Groq.",
            )

        with st.expander("Citation style"):
            st.session_state.citation_style = st.selectbox(
                "Style", options=["APA", "IEEE"], index=0,
                label_visibility="collapsed",
            )

        if st.session_state.papers:
            st.divider()
            st.subheader("Index Persistence")
            col_save, col_load = st.columns(2)
            with col_save:
                if st.button("💾 Save Index", use_container_width=True):
                    try:
                        st.session_state.store.save(INDEX_DIR)
                        st.success("Index saved.")
                    except Exception as e:
                        st.error(f"Save failed: {e}")
            with col_load:
                if st.button("📂 Load Index", use_container_width=True):
                    try:
                        st.session_state.store = FAISSVectorStore.load(INDEX_DIR)
                        st.success(
                            f"Loaded {st.session_state.store.total_chunks} chunks."
                        )
                    except FileNotFoundError:
                        st.warning("No saved index found.")
                    except Exception as e:
                        st.error(f"Load failed: {e}")

        if st.session_state.papers:
            st.divider()
            st.subheader("Quick Actions")
            paper_names = list(st.session_state.papers.keys())
            selected = st.selectbox(
                "Paper for shortcuts", paper_names,
                label_visibility="collapsed",
            )
            q_col1, q_col2 = st.columns(2)
            shortcuts = {
                "📝 Summarize": (
                    f"Summarize the paper '{selected}' covering its main "
                    "contributions, methodology, and key results."
                ),
                "🎯 Contributions": (
                    f"List the key contributions of the paper '{selected}'."
                ),
                "⚠️ Limitations": (
                    f"What limitations are discussed in the paper '{selected}'?"
                ),
                "🔬 Methodology": (
                    f"Describe the methodology used in the paper '{selected}'."
                ),
            }
            for i, (label, query_text) in enumerate(shortcuts.items()):
                col = q_col1 if i % 2 == 0 else q_col2
                if col.button(label, use_container_width=True):
                    st.session_state["_shortcut_query"] = query_text

        st.divider()
        if st.button(
            "🔄 Reset Session",
            type="secondary",
            use_container_width=True,
        ):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


def _chunk_confidence(chunk: Dict[str, Any]) -> str:
    from rag.grounding import compute_evidence_score, get_confidence_label
    score = compute_evidence_score([chunk])
    return get_confidence_label(score)


def _render_source_panel(
    chunks: List[Dict[str, Any]], title: str = "🔍 Source Chunks"
) -> None:
    if not chunks:
        return
    with st.expander(title):
        for i, chunk in enumerate(chunks, 1):
            rr = chunk.get("rerank_score")
            l2 = chunk.get("score")
            conf_label = _chunk_confidence(chunk)
            emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(conf_label, "⚪")

            col_meta, col_score = st.columns([3, 1])
            with col_meta:
                st.markdown(
                    f"**[{i}] {chunk['filename']}** — Page {chunk['page_number']}"
                )
            with col_score:
                st.markdown(f"{emoji} **{conf_label}**")

            score_parts = []
            if rr is not None:
                score_parts.append(f"Rerank: `{rr:.3f}`")
            if l2 is not None:
                score_parts.append(f"L2: `{l2:.3f}`")
            if score_parts:
                st.caption(" | ".join(score_parts))

            st.text_area(
                label="",
                value=chunk.get("text", ""),
                height=100,
                disabled=True,
                key=f"cp_{i}_{chunk['filename']}_{chunk['page_number']}_{title[:8]}",
                label_visibility="collapsed",
            )
            if i < len(chunks):
                st.divider()


def _render_rerank_comparison(
    raw_chunks: List[Dict[str, Any]],
    reranked_chunks: List[Dict[str, Any]],
) -> None:
    if not raw_chunks and not reranked_chunks:
        return
    with st.expander("⚡ Rerank Comparison (Before vs After)"):
        st.caption(
            "Left: FAISS retrieval order (L2 distance). "
            "Right: Cohere Rerank order (relevance score, higher = better)."
        )
        col_before, col_after = st.columns(2)
        with col_before:
            st.markdown("**Before Reranking**")
            for i, c in enumerate(raw_chunks[:8], 1):
                l2 = c.get("score", "?")
                l2_str = f"{l2:.3f}" if isinstance(l2, float) else str(l2)
                st.markdown(
                    f"{i}. `{c['filename']}` p.{c['page_number']} | L2: `{l2_str}`"
                )
        with col_after:
            st.markdown("**After Reranking**")
            for i, c in enumerate(reranked_chunks, 1):
                rr = c.get("rerank_score", "?")
                rr_str = f"{rr:.3f}" if isinstance(rr, float) else str(rr)
                st.markdown(
                    f"{i}. `{c['filename']}` p.{c['page_number']} | Score: `{rr_str}`"
                )


def _render_chat() -> None:
    st.header("💬 Ask Your Papers")

    if not st.session_state.papers:
        st.info(
            "👈 **Upload one or more research PDFs** in the sidebar, "
            "or click **Load Sample Paper** to try a demo instantly."
        )
        st.subheader("Sample questions you can ask:")
        for q in SAMPLE_QUESTIONS:
            st.markdown(f"- *{q}*")
        return

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                if msg.get("citations"):
                    with st.expander("📎 Citations & Sources"):
                        st.markdown(msg["citations"])
                        st.divider()
                        _render_formatted_citations(msg.get("used_chunks", []))
                if msg.get("used_chunks"):
                    _render_source_panel(
                        msg["used_chunks"],
                        title=f"🔍 Source Chunks [{msg.get('confidence_label', '')}]",
                    )

    shortcut = st.session_state.pop("_shortcut_query", None)
    user_input: Optional[str] = st.chat_input(
        "Ask a question about the uploaded papers …"
    )
    query: Optional[str] = shortcut or user_input

    if not query:
        return

    # ── Short-query guard ──────────────────────────────────────────────────────
    from utils.config import MIN_QUERY_WORDS
    if len(query.split()) < MIN_QUERY_WORDS:
        with st.chat_message("assistant"):
            st.info(
                f"💡 Your query **\"{query}\"** is too short to retrieve relevant "
                "passages. Try asking a full question, e.g.:\n"
                "- *What is the main contribution of this paper?*\n"
                "- *Summarize the methodology described in the paper.*\n"
                "- *What datasets were used in the experiments?*"
            )
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving relevant passages …"):
            raw_chunks = retrieve(
                query, st.session_state.store, top_k=st.session_state.top_k
            )

        if not raw_chunks:
            answer = (
                "I could not find any relevant content in the indexed papers. "
                "Try uploading additional papers or rephrasing your question."
            )
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            return

        with st.spinner("⚡ Reranking passages …"):
            reranked = rerank(query, raw_chunks, top_n=st.session_state.top_n)

        st.session_state.raw_chunks_last = raw_chunks
        st.session_state.reranked_last = reranked

        evidence_score, confidence_label, refuse = evaluate_grounding(reranked)

        if refuse:
            answer = build_refusal_message(evidence_score)
            st.markdown(answer)
            metrics = compute_query_metrics(
                query, raw_chunks, reranked, answer, was_refused=True
            )
            st.session_state.eval_history.append(metrics)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "confidence_label": confidence_label,
                    "evidence_score": evidence_score,
                    "used_chunks": reranked,
                }
            )
            return

        if st.session_state.use_streaming:
            placeholder = st.empty()
            full_answer = ""
            try:
                for token in stream_answer(query, reranked, chat_history=history):
                    full_answer += token
                    placeholder.markdown(full_answer + "▌")
                placeholder.markdown(full_answer)
                answer = full_answer
            except Exception:
                placeholder.empty()
                answer, _ = generate_answer(query, reranked, chat_history=history)
                st.markdown(answer)
        else:
            with st.spinner("🤖 Generating answer …"):
                answer, _ = generate_answer(query, reranked, chat_history=history)
            st.markdown(answer)

        annotated = annotate_answer(answer, evidence_score, confidence_label)
        conf_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(
            confidence_label, "⚪"
        )
        st.caption(
            f"{conf_emoji} **Confidence: {confidence_label}** "
            f"| Evidence score: `{evidence_score:.2f}`"
        )

        citations_text = format_citations(reranked)
        with st.expander("📎 Citations & Sources"):
            st.markdown(citations_text)
            st.divider()
            _render_formatted_citations(reranked)

        _render_source_panel(reranked, title=f"🔍 Source Chunks [{confidence_label}]")
        _render_rerank_comparison(raw_chunks, reranked)

    metrics = compute_query_metrics(
        query, raw_chunks, reranked, answer, was_refused=False
    )
    st.session_state.eval_history.append(metrics)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": annotated,
            "citations": citations_text,
            "used_chunks": reranked,
            "confidence_label": confidence_label,
            "evidence_score": evidence_score,
        }
    )


def _render_formatted_citations(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        return
    style = st.session_state.citation_style
    st.caption(f"Export format ({style}):")
    seen: set = set()
    idx = 1
    for chunk in chunks:
        key = (chunk["filename"], chunk["page_number"])
        if key in seen:
            continue
        seen.add(key)
        if style == "APA":
            st.code(build_apa_citation(chunk["filename"], chunk["page_number"]))
        else:
            st.code(
                build_ieee_citation(chunk["filename"], chunk["page_number"], idx)
            )
        idx += 1


def _render_compare() -> None:
    st.header("⚖️ Compare Two Papers")

    papers = list(st.session_state.papers.keys())
    if len(papers) < 2:
        st.info("Please upload **at least 2 papers** to use comparison mode.")
        return

    col_a, col_b = st.columns(2)
    with col_a:
        paper_a = st.selectbox("Paper A", papers, key="cmp_a")
    with col_b:
        remaining = [p for p in papers if p != paper_a]
        paper_b = st.selectbox("Paper B", remaining, key="cmp_b")

    st.divider()
    st.subheader("Quick Comparisons")
    quick = {
        "Methodology": "How do the methodologies of the two papers differ?",
        "Results": "Compare the results and findings of the two papers.",
        "Limitations": "What are the limitations identified in each paper?",
        "Contributions": "What are the unique contributions of each paper?",
    }
    btn_cols = st.columns(len(quick))
    selected_quick: Optional[str] = None
    for col, (label, q_text) in zip(btn_cols, quick.items()):
        if col.button(label, use_container_width=True):
            selected_quick = q_text

    st.divider()
    compare_query = st.text_area(
        "Custom comparison question:",
        value=selected_quick or "",
        placeholder="e.g. How do the two papers address class imbalance?",
        height=80,
    )
    run_btn = st.button(
        "⚖️ Compare Now", type="primary", use_container_width=True,
        disabled=not compare_query.strip(),
    )

    if not run_btn or not compare_query.strip():
        return

    with st.spinner("🔍 Running comparison pipeline …"):
        answer, chunks_a, chunks_b = compare_papers(
            query=compare_query,
            paper_a=paper_a,
            paper_b=paper_b,
            store=st.session_state.store,
            top_k=st.session_state.top_k,
            top_n=st.session_state.top_n,
        )

    all_chunks = chunks_a + chunks_b
    evidence_score, confidence_label, _ = evaluate_grounding(all_chunks)
    conf_emoji = {"High": "🟢", "Medium": "🟡", "Low": "🔴"}.get(
        confidence_label, "⚪"
    )

    st.divider()
    st.subheader("Comparison Result")
    st.caption(
        f"{conf_emoji} **Confidence: {confidence_label}** "
        f"| Evidence score: `{evidence_score:.2f}`"
    )
    st.markdown(answer)

    if chunks_a or chunks_b:
        src_a, src_b = st.columns(2)
        with src_a:
            st.subheader(f"Sources — {paper_a}")
            if chunks_a:
                _render_source_panel(chunks_a, title=f"🔍 Chunks — {paper_a}")
            else:
                st.caption("No relevant content found in Paper A.")
        with src_b:
            st.subheader(f"Sources — {paper_b}")
            if chunks_b:
                _render_source_panel(chunks_b, title=f"🔍 Chunks — {paper_b}")
            else:
                st.caption("No relevant content found in Paper B.")

    st.session_state.messages.extend(
        [
            {
                "role": "user",
                "content": f"[COMPARE] {paper_a} vs {paper_b}: {compare_query}",
            },
            {"role": "assistant", "content": answer},
        ]
    )


def _render_evaluation_dashboard() -> None:
    history = st.session_state.eval_history
    if not history:
        return

    with st.expander("📊 RAG Evaluation Dashboard", expanded=False):
        agg = aggregate_session_metrics(history)
        st.caption(
            f"Metrics across **{agg.get('total_queries', 0)}** queries in this session."
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric(
            "Hit Rate @ K",
            f"{agg.get('avg_hit_rate', 0):.0%}",
            help="Fraction of top-N chunks containing query keywords.",
        )
        col2.metric(
            "Context Precision",
            f"{agg.get('avg_context_precision', 0):.0%}",
            help="Fraction of chunks that cleared the minimum rerank score.",
        )
        col3.metric(
            "Groundedness",
            f"{agg.get('avg_groundedness_ratio', 0):.0%}",
            help="Fraction of chunks that clearly support the answer.",
        )
        col4.metric(
            "Fallback Rate",
            f"{agg.get('fallback_rate', 0):.0%}",
            help="Fraction of queries that triggered the soft refusal.",
        )

        col5, col6 = st.columns(2)
        col5.metric(
            "Avg Rerank Score",
            f"{agg.get('avg_mean_rerank_score', 0):.3f}",
        )
        col6.metric(
            "Rerank Improvement",
            f"{agg.get('avg_rerank_improvement', 0):+.3f}",
        )

        st.divider()
        st.caption("**Per-query breakdown (last 10):**")
        recent = history[-10:]
        rows = []
        for h in recent:
            rows.append(
                {
                    "Query": h["query"][:60] + ("…" if len(h["query"]) > 60 else ""),
                    "Hit Rate": f"{h['hit_rate_at_k']:.0%}",
                    "Precision": f"{h['context_precision']:.0%}",
                    "Rerank": f"{h['mean_rerank_score']:.3f}",
                    "Grounded": f"{h['groundedness_ratio']:.0%}",
                    "OK?": "✗" if h["was_refused"] else "✓",
                }
            )
        st.dataframe(rows, use_container_width=True)


def _render_references_panel() -> None:
    refs = st.session_state.references
    if not refs:
        return
    with st.expander("📖 Extracted References"):
        for fname, entries in refs.items():
            st.markdown(f"**{fname}** — {len(entries)} references")
            st.markdown(format_references_markdown(entries))
            st.divider()


def _render_export() -> None:
    if not st.session_state.messages:
        return

    st.divider()
    st.subheader("📥 Export Session")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    col_md, col_json = st.columns(2)

    with col_md:
        md_lines = [
            f"# {APP_TITLE} — Session Export",
            f"*Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}*",
            "",
        ]
        for msg in st.session_state.messages:
            role_label = "**You**" if msg["role"] == "user" else "**Assistant**"
            md_lines.append(f"### {role_label}")
            md_lines.append(msg["content"])
            if msg.get("citations"):
                md_lines.append(f"\n*{msg['citations']}*")
            if msg.get("confidence_label"):
                score = msg.get("evidence_score", 0)
                score_str = f"{score:.2f}" if isinstance(score, float) else str(score)
                md_lines.append(
                    f"*Confidence: {msg['confidence_label']} (score: {score_str})*"
                )
            md_lines.append("")

        for fname, entries in st.session_state.references.items():
            md_lines.append(f"## References — {fname}")
            md_lines.append(format_references_markdown(entries))
            md_lines.append("")

        st.download_button(
            label="📄 Download Chat (Markdown)",
            data="\n".join(md_lines),
            file_name=f"research_chat_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    with col_json:
        summary = {
            "exported_at": datetime.now().isoformat(),
            "papers": list(st.session_state.papers.keys()),
            "message_count": len(st.session_state.messages),
            "messages": [
                {
                    "role": m["role"],
                    "content": m["content"],
                    "confidence_label": m.get("confidence_label"),
                    "evidence_score": m.get("evidence_score"),
                }
                for m in st.session_state.messages
            ],
            "references": {
                fname: [r["text"] for r in entries]
                for fname, entries in st.session_state.references.items()
            },
            "evaluation_summary": aggregate_session_metrics(
                st.session_state.eval_history
            ),
        }
        st.download_button(
            label="📊 Download Summary (JSON)",
            data=json.dumps(summary, indent=2, ensure_ascii=False),
            file_name=f"research_summary_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
        )


def main() -> None:
    _render_sidebar()

    _check_api_keys()  # Will call st.stop() if keys are missing

    if st.session_state.mode == "chat":
        _render_chat()
    else:
        _render_compare()

    _render_evaluation_dashboard()
    _render_references_panel()
    _render_export()


if __name__ == "__main__":
    main()
