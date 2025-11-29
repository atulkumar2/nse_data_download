"""Semantic search utilities over a ChromaDB collection of PDF chunks.

Lazily initializes the Chroma collection and embedding model on first use.
The persist directory can be overridden via the PERSIST_DIR environment
variable or by calling init_search_resources().
"""

import logging
import os
from types import SimpleNamespace
from typing import cast

import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_PERSIST_DIR = './vector-data'
DEFAULT_COLLECTION = 'pdf_chunks'
DEFAULT_MODEL = 'intfloat/multilingual-e5-base'

_state = SimpleNamespace(collections=[], model=None)
logger = logging.getLogger("semantic_search")


def load_persisted_collection(
    collection_name: str = DEFAULT_COLLECTION,
    persist_dir: str = DEFAULT_PERSIST_DIR,
):
    """Load a single ChromaDB collection from one persistent directory."""
    persistent_client = chromadb.PersistentClient(path=persist_dir)
    collection = persistent_client.get_collection(collection_name)
    logger.info(
        "Loaded collection '%s' from '%s' (count=%d)",
        collection_name,
        persist_dir,
        collection.count(),
    )
    return collection


def load_persisted_collections(
    collection_name: str = DEFAULT_COLLECTION,
    persist_dirs: list[str] | None = None,
):
    """Load the same-named collection from multiple persistent directories.

    Returns list of Collection objects. Directories that fail to load are
    logged and skipped.
    """
    if not persist_dirs:
        persist_dirs = [DEFAULT_PERSIST_DIR]
    collections = []
    for d in persist_dirs:
        try:
            col = load_persisted_collection(
                collection_name=collection_name,
                persist_dir=d,
            )
            collections.append(col)
        except Exception as e:  # noqa: BLE001
            logger.warning("Skipping '%s' due to load error: %s", d, e)
    if not collections:
        raise RuntimeError("No collections loaded from provided directories")
    return collections

def init_search_resources(
    persist_dir: str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    model_name: str = DEFAULT_MODEL,
) -> None:
    """Initialize and cache collections + embedding model.

    persist_dir may specify multiple directories separated by commas or
    os.pathsep (':').
    """
    if persist_dir is None:
        persist_dir = os.environ.get("PERSIST_DIR", DEFAULT_PERSIST_DIR)
    # Split on comma and os.pathsep, strip empties
    raw_parts = []
    for segment in persist_dir.split(","):
        raw_parts.extend(segment.split(os.pathsep))
    dirs = [p.strip() for p in raw_parts if p.strip()]
    if not dirs:
        dirs = [DEFAULT_PERSIST_DIR]
    if len(dirs) == 1:
        _state.collections = [
            load_persisted_collection(
                collection_name=collection_name,
                persist_dir=dirs[0],
            )
        ]
    else:
        logger.info(
            "Loading multiple collections from %d directories", len(dirs)
        )
        _state.collections = load_persisted_collections(
            collection_name=collection_name,
            persist_dirs=dirs,
        )
    _state.model = SentenceTransformer(model_name)


def semantic_search(
        query: str,
        n_results: int = 5,
        min_similarity: float = 0.1,
) -> list:
    """
    Performs semantic search on the ChromaDB collection and returns matching documents.

    This function can be used as a tool by the Gemini agent to search through
    the ingested PDF documents.

    Args:
        query (str): The search query string.
        n_results (int, optional): Number of results to return. Defaults to 5.
        min_similarity (float, optional): Minimum similarity threshold (0-1).

    Returns:
        list: List of dictionaries, each containing:
            - 'document' (str): The document text content
            - 'metadata' (dict): Metadata including source file and chunk index
            - 'similarity' (float): Similarity score (0-1, higher is better)
    """
    # Ensure resources are initialized
    # Ensure resources are initialized
    if (not _state.collections) or _state.model is None:
        init_search_resources()

    # Add query prefix required by multilingual-e5-base model
    prefixed_query = f"query: {query}"
    query_embedding = _state.model.encode(prefixed_query)  # type: ignore[union-attr]
    # Convert to plain list[float] if needed for Chroma types
    if hasattr(query_embedding, 'tolist'):
        query_embedding = query_embedding.tolist()
    qe_list = cast(list[float], query_embedding)

    # Perform semantic search
    aggregate_results = []
    for col in _state.collections:  # type: ignore[attr-defined]
        col_results = col.query(query_embeddings=[qe_list], n_results=n_results)
        if not col_results or not col_results.get("documents"):
            continue
        documents = col_results["documents"][0]  # type: ignore[index]
        metadatas = col_results["metadatas"][0]  # type: ignore[index]
        scores_or_distances = col_results.get("distances",
                  col_results.get("scores", [[]]))[0]  # type: ignore[index]
        for doc, meta, score in zip(
            documents,
            metadatas,
            scores_or_distances,
        ):
            if "scores" in col_results:
                similarity = score
            elif "distances" in col_results:
                similarity = 1 - score
            else:
                similarity = None
            if similarity is not None and similarity >= min_similarity:
                # Optionally add origin info at ingest time if required.
                aggregate_results.append(
                    {
                        "document": doc,
                        "metadata": meta,
                        "similarity": round(similarity, 4),
                    }
                )
    # Sort combined results by similarity desc and truncate
    aggregate_results.sort(key=lambda r: r["similarity"], reverse=True)
    return aggregate_results[:n_results]

if __name__ == '__main__':
    # Configure logging only for direct script execution
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s %(name)s: %(message)s',
        )

    init_search_resources()
    # Example usage: Interactive semantic search
    while True:
        query_text = input("Enter your search query (or type 'exit' to quit): ").strip()
        if query_text.lower() in ('exit', 'quit'):
            logger.info("Exiting semantic search.")
            break

        logger.info("Performing semantic search for: %s", query_text)
        results = semantic_search(query_text, n_results=3, min_similarity=0.1)

        if results:
            logger.info("Found %d results:", len(results))
            for i, result in enumerate(results, 1):
                logger.info(
                    "Result %d (similarity: %.4f)", i, result['similarity']
                )
                logger.info(
                    "Source: %s",
                    result['metadata'].get('source', 'Unknown'),
                )
                logger.info("Content: %s...", result['document'][:300])
        else:
            logger.info("No results found matching the query.")
