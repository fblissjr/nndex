"""Search result formatting."""

from typing import Dict, List

import numpy as np

from .chunker import Chunk


def format_results(
    chunks: List[Chunk],
    indices: np.ndarray,
    similarities: np.ndarray,
) -> List[Dict]:
    """Format search results with chunk metadata.

    Args:
        chunks: All indexed chunks.
        indices: Array of chunk indices from search.
        similarities: Array of similarity scores.

    Returns:
        List of result dicts with file, lines, similarity, preview.
    """
    results = []
    for idx, sim in zip(indices, similarities):
        chunk = chunks[int(idx)]
        results.append({
            "file": chunk.file_path,
            "lines": f"{chunk.start_line}-{chunk.end_line}",
            "similarity": float(sim),
            "preview": chunk.preview,
        })
    return results
