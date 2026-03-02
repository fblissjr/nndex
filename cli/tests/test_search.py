"""Tests for search and index persistence."""

import pytest
import numpy as np
from pathlib import Path
from nndex_cli.chunker import Chunk


class TestBuildIndex:
    """Test building an nndex index from embeddings."""

    def test_build_and_search(self):
        from nndex_cli.index import build_index, search_index

        # 3 chunks with 4-dim embeddings
        chunks = [
            Chunk("a.py", 1, 5, "authentication logic", "authentication logic"),
            Chunk("b.py", 1, 3, "database connection", "database connection"),
            Chunk("c.py", 1, 2, "user login handler", "user login handler"),
        ]
        embeddings = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],  # Similar to first
        ], dtype=np.float32)

        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        index = build_index(embeddings)
        assert index is not None

        # Search with query similar to first embedding
        query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        query = query / np.linalg.norm(query)
        indices, similarities = search_index(index, query, k=2)

        assert len(indices) == 2
        # First result should be index 0 (exact match)
        assert indices[0] == 0
        # Second should be index 2 (similar)
        assert indices[1] == 2


class TestPersistence:
    """Test save/load round-trip for index and metadata."""

    def test_save_and_load_metadata(self, tmp_path):
        from nndex_cli.index import save_metadata, load_metadata

        chunks = [
            Chunk("a.py", 1, 5, "content a", "preview a"),
            Chunk("b.py", 10, 20, "content b", "preview b"),
        ]

        nndex_dir = tmp_path / ".nndex"
        save_metadata(nndex_dir, chunks, model_name="test-model", dims=768)

        loaded_chunks, meta = load_metadata(nndex_dir)
        assert len(loaded_chunks) == 2
        assert loaded_chunks[0].file_path == "a.py"
        assert loaded_chunks[0].start_line == 1
        assert loaded_chunks[0].preview == "preview a"
        assert loaded_chunks[1].file_path == "b.py"
        assert meta["model"] == "test-model"
        assert meta["dims"] == 768

    def test_save_and_load_embeddings(self, tmp_path):
        from nndex_cli.index import save_embeddings, load_embeddings

        embeddings = np.random.randn(10, 32).astype(np.float32)
        nndex_dir = tmp_path / ".nndex"
        save_embeddings(nndex_dir, embeddings)

        loaded = load_embeddings(nndex_dir)
        assert loaded.shape == (10, 32)
        np.testing.assert_array_almost_equal(loaded, embeddings)


class TestSearchResults:
    """Test formatting search results with chunk metadata."""

    def test_format_results(self):
        from nndex_cli.search import format_results

        chunks = [
            Chunk("auth.py", 1, 10, "def login():\n    pass", "def login(): pass"),
            Chunk("db.py", 5, 15, "def connect():\n    pass", "def connect(): pass"),
            Chunk("auth.py", 20, 30, "def logout():\n    pass", "def logout(): pass"),
        ]
        indices = np.array([0, 2])
        similarities = np.array([0.95, 0.82])

        results = format_results(chunks, indices, similarities)
        assert len(results) == 2
        assert results[0]["file"] == "auth.py"
        assert results[0]["lines"] == "1-10"
        assert results[0]["similarity"] == pytest.approx(0.95, abs=0.01)
        assert results[0]["preview"] == "def login(): pass"
        assert results[1]["file"] == "auth.py"
        assert results[1]["lines"] == "20-30"
