"""Embedding logic using MLXEmbeddingProvider."""

from typing import List, Optional

from .chunker import Chunk


def embed_chunks(
    chunks: List[Chunk],
    provider,
    task: str = "document",
    batch_size: int = 32,
) -> List[List[float]]:
    """Embed a list of chunks using the provider.

    Args:
        chunks: Text chunks to embed.
        provider: An MLXEmbeddingProvider (or any object with get_embeddings).
        task: Task prefix type ("document" for indexing, "query" for search).
        batch_size: Number of texts per batch.

    Returns:
        List of embedding vectors (one per chunk).
    """
    all_embeddings = []
    texts = [c.content for c in chunks]

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = provider.get_embeddings(batch, task=task)
        all_embeddings.extend(batch_embeddings)

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
