"""
Embedding generation via the Cohere Embed API.

Two public functions are provided:
    embed_documents(texts) — for indexing passages (input_type="search_document")
    embed_query(query)     — for query lookup  (input_type="search_query")

Using the correct input_type is essential: Cohere trains separate
representations for documents and queries.  Mixing them degrades recall.

Batching respects Cohere's maximum of 96 texts per request.
"""

from typing import List

import numpy as np
import cohere

from utils.logger import get_logger
from utils import config

logger = get_logger(__name__)

# Module-level client cache.  Invalidated when the API key changes.
_client: cohere.Client | None = None
_client_key: str = ""


def _get_client() -> cohere.Client:
    """
    Return a Cohere client, creating or recreating it when the API key
    has changed (e.g. the user updated .env while the app was running).
    """
    global _client, _client_key
    # Re-read from the environment every call so key updates are picked up
    # without restarting the app.
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


def embed_documents(
    texts: List[str],
    batch_size: int = 96,
) -> np.ndarray:
    """
    Embed a list of document passages for storage in the vector index.

    Uses input_type="search_document" — the correct representation for
    passages that will be retrieved (not queried).

    Args:
        texts:      Non-empty list of strings to embed.
        batch_size: Texts per API call (Cohere max is 96).

    Returns:
        Float32 NumPy array of shape (len(texts), COHERE_EMBED_DIM).

    Raises:
        ValueError: If *texts* is empty.
    """
    if not texts:
        raise ValueError("embed_documents: received an empty list of texts.")

    client = _get_client()
    all_embeddings: List[List[float]] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(
            f"Embedding batch {batch_num}/{total_batches} "
            f"({len(batch)} text(s), input_type=search_document)"
        )

        response = client.embed(
            texts=batch,
            model=config.COHERE_EMBED_MODEL,
            input_type="search_document",
        )
        all_embeddings.extend(response.embeddings)

    arr = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"Finished embedding {len(texts)} document(s) → shape {arr.shape}")
    return arr


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single user query for nearest-neighbour search.

    Uses input_type="search_query" — the correct representation for
    lookup queries (not for stored passages).

    Args:
        query: Non-empty query string.

    Returns:
        Float32 NumPy array of shape (1, COHERE_EMBED_DIM).

    Raises:
        ValueError: If *query* is empty or whitespace-only.
    """
    if not query or not query.strip():
        raise ValueError("embed_query: query cannot be empty.")

    client = _get_client()

    response = client.embed(
        texts=[query.strip()],
        model=config.COHERE_EMBED_MODEL,
        input_type="search_query",
    )

    arr = np.array(response.embeddings, dtype=np.float32)
    logger.info(f"Embedded query → shape {arr.shape}")
    return arr
