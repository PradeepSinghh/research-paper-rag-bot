"""
Reranking via the Cohere Rerank API.

FAISS retrieval (approximate nearest-neighbour in embedding space) can
return chunks that are topically adjacent but not the most relevant for
the exact phrasing of a user's question.  Cohere Rerank applies a
cross-encoder model that scores each (query, chunk) pair jointly,
producing a significant precision boost with minimal latency overhead.

Usage pattern:
    raw_chunks  = retrieve(query, store, top_k=10)
    best_chunks = rerank(query, raw_chunks, top_n=5)
"""

from typing import List, Dict, Any

import cohere

from utils.logger import get_logger
from utils import config

logger = get_logger(__name__)

_client: cohere.Client | None = None
_client_key: str = ""


def _get_client() -> cohere.Client:
    """
    Return a Cohere client, creating or recreating it when the API key
    has changed (e.g. the user updated .env while the app was running).
    """
    global _client, _client_key
    from dotenv import load_dotenv
    import os
    load_dotenv(override=True)
    api_key = os.getenv("COHERE_API_KEY", "")
    if not api_key:
        raise ValueError(
            "COHERE_API_KEY is not set. "
            "Add it to your .env file and restart the app."
        )
    if _client is None or api_key != _client_key:
        _client = cohere.Client(api_key=api_key)
        _client_key = api_key
    return _client


def rerank(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: int = config.TOP_K_RERANK,
) -> List[Dict[str, Any]]:
    """
    Reorder *chunks* by their relevance to *query* using Cohere Rerank.

    Args:
        query:  The user's question.
        chunks: Candidate chunk dicts from retrieve().  Must have a ``text`` key.
        top_n:  Maximum number of chunks to keep after reranking.

    Returns:
        List of chunk dicts (length ≤ top_n) sorted best-first by
        ``rerank_score`` (float in [0, 1]; higher = more relevant).
        Returns an empty list if *chunks* is empty.
    """
    if not chunks:
        logger.warning("rerank() called with an empty chunks list — returning [].")
        return []

    top_n = min(top_n, len(chunks))
    client = _get_client()

    documents = [c["text"] for c in chunks]

    logger.info(
        f"Reranking {len(documents)} chunk(s) → keeping top {top_n} "
        f"| query='{query[:60]}'"
    )

    response = client.rerank(
        query=query,
        documents=documents,
        model=config.COHERE_RERANK_MODEL,
        top_n=top_n,
    )

    reranked: List[Dict[str, Any]] = []
    for result in response.results:
        chunk = dict(chunks[result.index])          # copy — don't mutate original
        chunk["rerank_score"] = round(result.relevance_score, 4)
        reranked.append(chunk)

    scores = [c["rerank_score"] for c in reranked]
    logger.info(f"Rerank scores (best first): {scores}")

    return reranked
