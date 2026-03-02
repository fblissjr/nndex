"""Embedding logic using MLXEmbeddingProvider."""

from typing import Callable, List, Optional

from .chunker import Chunk


def embed_chunks(
    chunks: List[Chunk],
    provider,
    task: str = "document",
    batch_size: int = 32,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> List[List[float]]:
    """Embed a list of chunks using the provider.

    Args:
        chunks: Text chunks to embed.
        provider: An MLXEmbeddingProvider (or any object with get_embeddings).
        task: Task prefix type ("document" for indexing, "query" for search).
        batch_size: Number of texts per batch.
        on_progress: Optional callback(done, total) called after each batch.

    Returns:
        List of embedding vectors (one per chunk).
    """
    all_embeddings = []
    texts = [c.content for c in chunks]
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = provider.get_embeddings(batch, task=task)
        all_embeddings.extend(batch_embeddings)
        if on_progress is not None:
            on_progress(min(i + len(batch), total), total)

    return all_embeddings


def embed_query(query: str, provider) -> List[float]:
    """Embed a single query string.

    Uses "query" task prefix for optimal retrieval quality.

    Args:
        query: The search query text.
        provider: An MLXEmbeddingProvider.

    Returns:
        Embedding vector as a list of floats.
    """
    results = provider.get_embeddings([query], task="query")
    return results[0]
