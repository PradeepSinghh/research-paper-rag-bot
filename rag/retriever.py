"""
Retrieval — embed a query and fetch the top-k nearest chunks from FAISS.

This thin module exists to keep the retrieval step clearly separated from
embedding and reranking so each stage can be tested independently.
"""

from typing import List, Dict, Any, Optional

from rag.embeddings import embed_query
from rag.vectorstore import FAISSVectorStore
from utils.logger import get_logger
from utils import config

logger = get_logger(__name__)


def retrieve(
    query: str,
    store: FAISSVectorStore,
    top_k: int = config.TOP_K_RETRIEVE,
    paper_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve the *top_k* most relevant chunks for *query*.

    Steps:
    1. Embed *query* with Cohere (input_type="search_query").
    2. Run nearest-neighbour search against the FAISS index.
    3. Optionally filter results to a single paper (*paper_filter*).

    Args:
        query:        User's question (non-empty string).
        store:        Populated FAISSVectorStore instance.
        top_k:        Number of chunks to retrieve before reranking.
        paper_filter: If set, only return chunks whose ``filename`` matches
                      this value.  Useful for single-paper Q&A.

    Returns:
        List of chunk dicts sorted best-first (ascending L2 distance).
        Each dict has a ``score`` key (L2 distance).  May be empty if
        the store is empty or the filter yields no results.
    """
    logger.info(
        f"Retrieving top {top_k} chunk(s) | filter='{paper_filter}' "
        f"| query='{query[:80]}'"
    )

    query_embedding = embed_query(query)
    results = store.search(query_embedding, top_k=top_k)

    if paper_filter:
        before = len(results)
        results = [r for r in results if r.get("filename") == paper_filter]
        logger.info(
            f"paper_filter='{paper_filter}' kept {len(results)}/{before} chunk(s)"
        )

    logger.info(f"Retrieved {len(results)} chunk(s)")
    return results
