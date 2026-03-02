"""Tests for embedding logic."""

import pytest
from unittest.mock import MagicMock, patch
from nndex_cli.chunker import Chunk


class TestEmbedChunks:
    """Test embedding chunks via the provider."""

    def test_embed_single_chunk(self):
        from nndex_cli.embed import embed_chunks

        mock_provider = MagicMock()
        mock_provider.get_embeddings.return_value = [[0.1, 0.2, 0.3]]

        chunks = [Chunk(
            file_path="test.py",
            start_line=1,
            end_line=5,
            content="def hello(): pass",
            preview="def hello(): pass",
        )]

        embeddings = embed_chunks(chunks, mock_provider, task="document")
        assert len(embeddings) == 1
        assert embeddings[0] == [0.1, 0.2, 0.3]
        mock_provider.get_embeddings.assert_called_once()

    def test_embed_batched(self):
        from nndex_cli.embed import embed_chunks

        mock_provider = MagicMock()
        # Two batches: first has 2 chunks, second has 1
        mock_provider.get_embeddings.side_effect = [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6]],
        ]

        chunks = [
            Chunk("a.py", 1, 1, "a", "a"),
            Chunk("b.py", 1, 1, "b", "b"),
            Chunk("c.py", 1, 1, "c", "c"),
        ]

        embeddings = embed_chunks(chunks, mock_provider, task="document", batch_size=2)
        assert len(embeddings) == 3
        assert mock_provider.get_embeddings.call_count == 2

    def test_embed_query(self):
        from nndex_cli.embed import embed_query

        mock_provider = MagicMock()
        mock_provider.get_embeddings.return_value = [[1.0, 0.0, 0.0]]

        result = embed_query("find auth", mock_provider)
        assert result == [1.0, 0.0, 0.0]
        # Query should use "query" task prefix
        mock_provider.get_embeddings.assert_called_once_with(
            ["find auth"], task="query"
        )
