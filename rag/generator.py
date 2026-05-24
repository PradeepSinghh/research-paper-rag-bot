"""
Answer generation using the Groq LLM API.

The LLM receives:
  • A system prompt enforcing grounded, citation-tagged answers.
  • Numbered context blocks built from the reranked chunks.
  • The user's question.
  • (Optionally) recent chat history for multi-turn conversations.

The model is instructed to cite sources in the format
[Source: <filename>, Page <N>] so citations can be parsed downstream.
"""

from typing import Generator, List, Dict, Any, Tuple, Optional

from groq import Groq

from utils.logger import get_logger
from utils import config

logger = get_logger(__name__)

_client: Optional[Groq] = None
_client_key: str = ""

# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise and helpful research assistant. "
    "You answer questions based ONLY on the provided research paper excerpts.\n\n"
    "Rules:\n"
    "1. Always ground your answer in the provided context.\n"
    "2. Cite every factual claim using the format [Source: <filename>, Page <N>].\n"
    "3. If the answer is not present in the context, respond with: "
    "'I could not find information about this in the provided papers.'\n"
    "4. Do NOT speculate, hallucinate, or use knowledge outside the context.\n"
    "5. Be concise and structured. Use bullet points or numbered lists when helpful.\n"
    "6. When multiple papers are referenced, clearly attribute each claim."
)


# ── Internal helpers (also used by compare.py) ─────────────────────────────────

def get_groq_client() -> Groq:
    """
    Return a Groq client, creating or recreating it when the API key
    has changed (e.g. the user updated .env while the app was running).
    """
    global _client, _client_key
    from dotenv import load_dotenv
    import os
    load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file and restart the app."
        )
    if _client is None or api_key != _client_key:
        _client = Groq(api_key=api_key)
        _client_key = api_key
    return _client


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """
    Format a list of chunk dicts into a numbered, labelled context block.

    Each entry is prefixed with its 1-based index so the LLM can reference
    it accurately in citations.

    Args:
        chunks: Reranked chunk dicts (must have ``text``, ``filename``,
                ``page_number`` keys).

    Returns:
        Multi-line context string.
    """
    parts: List[str] = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[{i}] Source: {chunk['filename']}, Page {chunk['page_number']}"
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = config.GROQ_TEMPERATURE,
    max_tokens: int = config.GROQ_MAX_TOKENS,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate a grounded answer using the Groq LLM.

    Args:
        query:        The user's question.
        chunks:       Reranked chunks to use as evidence context.
        chat_history: Previous messages in OpenAI format
                      [{"role": "user"|"assistant", "content": "..."}].
                      The last 8 messages are included (4 turns) to stay
                      within the model's context window budget.
        temperature:  Sampling temperature (lower = more deterministic).
        max_tokens:   Maximum tokens in the generated response.

    Returns:
        Tuple of (answer_text, used_chunks).
        ``used_chunks`` is the same list that was passed in, returned so the
        caller can build citations without a separate variable.
    """
    if not chunks:
        logger.warning("generate_answer called with no chunks — returning fallback.")
        return (
            "I could not find any relevant information in the uploaded papers.",
            [],
        )

    client = get_groq_client()
    context = build_context(chunks)

    user_message = (
        f"Context from research papers:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer strictly from the context above. "
        "Cite sources using [Source: filename, Page N]."
    )

    # Build message list: system → history → current user turn
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        # Include at most the 4 most recent exchange pairs (8 messages)
        messages.extend(chat_history[-8:])

    messages.append({"role": "user", "content": user_message})

    logger.info(
        f"Calling Groq [{config.GROQ_MODEL}] | "
        f"{len(chunks)} context chunk(s) | query='{query[:60]}'"
    )

    response = client.chat.completions.create(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    answer: str = response.choices[0].message.content or ""
    logger.info(f"Generated answer: {len(answer)} character(s)")

    return answer, chunks


def stream_answer(
    query: str,
    chunks: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
    temperature: float = config.GROQ_TEMPERATURE,
    max_tokens: int = config.GROQ_MAX_TOKENS,
) -> Generator[str, None, None]:
    """
    Stream answer tokens from the Groq LLM one chunk at a time.

    Yields successive text deltas.  The caller is responsible for
    accumulating them into the final answer string.

    Args:
        query:        The user's question.
        chunks:       Reranked chunks to use as evidence context.
        chat_history: Optional prior conversation messages.
        temperature:  Sampling temperature.
        max_tokens:   Maximum tokens to generate.

    Yields:
        Successive text fragments (str) as they arrive from the API.
    """
    if not chunks:
        yield "I could not find any relevant information in the uploaded papers."
        return

    client = get_groq_client()
    context = build_context(chunks)

    user_message = (
        f"Context from research papers:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer strictly from the context above. "
        "Cite sources using [Source: filename, Page N]."
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        messages.extend(chat_history[-8:])
    messages.append({"role": "user", "content": user_message})

    logger.info(
        f"Streaming Groq [{config.GROQ_MODEL}] | "
        f"{len(chunks)} chunk(s) | query='{query[:60]}'"
    )

    with client.chat.completions.stream(
        model=config.GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    ) as stream:
        for text in stream.text_stream:
            yield text
