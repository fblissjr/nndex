"""nndex CLI -- embed, index, and search code with natural language."""

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_MODEL_PATH = os.environ.get(
    "NNDEX_MODEL_PATH",
    "/Users/fredbliss/Storage/google_embeddinggemma-300m",
)
DEFAULT_API_URL = os.environ.get("NNDEX_API_URL", "http://localhost:8000")


def cmd_embed(args):
    """Embed files in a directory and build an index."""
    from .chunker import chunk_directory
    from .embed import embed_chunks
    from .index import save_embeddings, save_metadata

    root = Path(args.directory).resolve()
    nndex_dir = root / ".nndex"
    model_path = args.model

    print(f"chunking files in {root} ...")
    chunks = chunk_directory(root)
    if not chunks:
        print("no text files found.")
        return

    print(f"found {len(chunks)} chunks across {len({c.file_path for c in chunks})} files")

    # Load embedding provider
    print(f"loading model from {model_path} ...")
    t0 = time.time()
    provider = _load_provider(model_path)
    print(f"model loaded in {time.time() - t0:.1f}s")

    # Embed
    t0 = time.time()

    def _progress(done, total):
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        print(f"\r  embedding: {done}/{total} chunks ({rate:.0f}/s)", end="", flush=True)

    embeddings_list = embed_chunks(chunks, provider, task="document",
                                   on_progress=_progress)
    embeddings = np.array(embeddings_list, dtype=np.float32)
    elapsed = time.time() - t0
    print(f"\r  embedded {len(chunks)} chunks in {elapsed:.1f}s ({len(chunks)/elapsed:.0f} chunks/s)")

    # Save
    save_embeddings(nndex_dir, embeddings)
    save_metadata(nndex_dir, chunks, model_name=model_path, dims=embeddings.shape[1])
    print(f"index saved to {nndex_dir}")


def cmd_search(args):
    """Search the index with a natural language query."""
    from .embed import embed_query
    from .index import build_index, load_embeddings, load_metadata, search_index
    from .search import format_results

    root = Path(args.directory).resolve()
    nndex_dir = root / ".nndex"

    if not (nndex_dir / "metadata.json").exists():
        print(f"no index found at {nndex_dir}. run 'nndex embed' first.")
        sys.exit(1)

    # Load metadata and embeddings
    chunks, meta = load_metadata(nndex_dir)
    embeddings = load_embeddings(nndex_dir)
    model_path = meta["model"]

    # Build in-memory index
    index = build_index(embeddings, normalized=True)

    # Load provider and embed query
    provider = _load_provider(model_path)
    query_vec = embed_query(args.query, provider)
    query_arr = np.array(query_vec, dtype=np.float32)

    # Search
    indices, similarities = search_index(index, query_arr, k=args.k)
    results = format_results(chunks, indices, similarities)

    # Display
    print()
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['file']}:{r['lines']}  (sim={r['similarity']:.3f})")
        print(f"     {r['preview']}")
        print()


def cmd_info(args):
    """Print index stats."""
    from .index import load_metadata

    root = Path(args.directory).resolve()
    nndex_dir = root / ".nndex"

    if not (nndex_dir / "metadata.json").exists():
        print(f"no index found at {nndex_dir}.")
        return

    _, meta = load_metadata(nndex_dir)
    print(f"  vectors: {meta['num_chunks']}")
    print(f"  dimensions: {meta['dims']}")
    print(f"  model: {meta['model']}")


def _try_load_mlx_provider(model_path: str):
    """Try to load MLXEmbeddingProvider. Returns None if heylookitsanllm is not installed."""
    try:
        from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider
    except ImportError:
        return None

    provider = MLXEmbeddingProvider(
        model_id="embeddinggemma",
        config={"model_path": model_path},
        verbose=False,
    )
    provider.load_model()
    return provider


def _load_provider_auto(
    model_path: str = DEFAULT_MODEL_PATH,
    api_url: str = DEFAULT_API_URL,
):
    """Load embedding provider, preferring local MLX, falling back to HTTP.

    Args:
        model_path: Path to local model directory (for MLX provider).
        api_url: URL of embeddings API server (for HTTP fallback).

    Returns:
        An object with get_embeddings(texts, task=...) method.
    """
    provider = _try_load_mlx_provider(model_path)
    if provider is not None:
        return provider

    from .http_provider import HTTPEmbeddingProvider
    return HTTPEmbeddingProvider(api_url=api_url)


def _load_provider(model_path: str):
    """Load MLXEmbeddingProvider for the given model path."""
    from heylook_llm.providers.mlx_embedding_provider import MLXEmbeddingProvider

    provider = MLXEmbeddingProvider(
        model_id="embeddinggemma",
        config={"model_path": model_path},
        verbose=False,
    )
    provider.load_model()
    return provider


def main():
    parser = argparse.ArgumentParser(
        prog="nndex",
        description="embed, index, and search code with natural language",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # embed
    p_embed = subparsers.add_parser("embed", help="embed files and build index")
    p_embed.add_argument("directory", nargs="?", default=".", help="directory to index")
    p_embed.add_argument("--model", default=DEFAULT_MODEL_PATH, help="model path or HF repo")
    p_embed.set_defaults(func=cmd_embed)

    # search
    p_search = subparsers.add_parser("search", help="search the index")
    p_search.add_argument("query", help="natural language search query")
    p_search.add_argument("--directory", "-d", default=".", help="directory with .nndex index")
    p_search.add_argument("--k", type=int, default=10, help="number of results")
    p_search.set_defaults(func=cmd_search)

    # info
    p_info = subparsers.add_parser("info", help="print index stats")
    p_info.add_argument("directory", nargs="?", default=".", help="directory with .nndex index")
    p_info.set_defaults(func=cmd_info)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
