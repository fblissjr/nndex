"""Index building, persistence, and search using nndex."""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import orjson

from .chunker import Chunk


def build_index(embeddings: np.ndarray, normalized: bool = True):
    """Build an nndex index from embeddings.

    Args:
        embeddings: (n, dims) float32 array of embeddings.
        normalized: Whether embeddings are already L2-normalized.

    Returns:
        An nndex.NNdex instance.
    """
    from nndex import NNdex

    return NNdex(embeddings, normalized=normalized)


def search_index(
    index,
    query: np.ndarray,
    k: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Search the index with a query vector.

    Args:
        index: An nndex.NNdex instance.
        query: (dims,) float32 query vector.
        k: Number of results to return.

    Returns:
        (indices, similarities) arrays.
    """
    indices, similarities = index.search(query, k=k)
    return indices, similarities


def save_metadata(
    nndex_dir: Path,
    chunks: List[Chunk],
    model_name: str,
    dims: int,
):
    """Save chunk metadata to .nndex/metadata.json.

    Args:
        nndex_dir: Path to .nndex directory.
        chunks: List of indexed chunks.
        model_name: Model used for embeddings.
        dims: Embedding dimensions.
    """
    nndex_dir.mkdir(parents=True, exist_ok=True)

    data = {
        "model": model_name,
        "dims": dims,
        "num_chunks": len(chunks),
        "chunks": [
            {
                "file_path": c.file_path,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "preview": c.preview,
            }
            for c in chunks
        ],
    }

    metadata_path = nndex_dir / "metadata.json"
    metadata_path.write_bytes(orjson.dumps(data, option=orjson.OPT_INDENT_2))


def load_metadata(nndex_dir: Path) -> Tuple[List[Chunk], Dict]:
    """Load chunk metadata from .nndex/metadata.json.

    Args:
        nndex_dir: Path to .nndex directory.

    Returns:
        (chunks, meta_dict) where meta_dict has "model", "dims", etc.
    """
    metadata_path = nndex_dir / "metadata.json"
    data = orjson.loads(metadata_path.read_bytes())

    chunks = [
        Chunk(
            file_path=c["file_path"],
            start_line=c["start_line"],
            end_line=c["end_line"],
            content="",  # Content not stored in metadata
            preview=c["preview"],
        )
        for c in data["chunks"]
    ]

    meta = {
        "model": data["model"],
        "dims": data["dims"],
        "num_chunks": data["num_chunks"],
    }

    return chunks, meta


def save_embeddings(nndex_dir: Path, embeddings: np.ndarray):
    """Save embeddings to .nndex/embeddings.npy.

    Args:
        nndex_dir: Path to .nndex directory.
        embeddings: (n, dims) float32 array.
    """
    nndex_dir.mkdir(parents=True, exist_ok=True)
    np.save(nndex_dir / "embeddings.npy", embeddings)


def load_embeddings(nndex_dir: Path) -> np.ndarray:
    """Load embeddings from .nndex/embeddings.npy.

    Args:
        nndex_dir: Path to .nndex directory.

    Returns:
        (n, dims) float32 array.
    """
    return np.load(nndex_dir / "embeddings.npy")
