"""Tests for embedding logic."""

import json

import pytest
from unittest.mock import MagicMock, patch, Mock
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

    def test_progress_callback_not_called_when_none(self):
        """embed_chunks should work without a progress callback."""
        from nndex_cli.embed import embed_chunks

        mock_provider = MagicMock()
        mock_provider.get_embeddings.return_value = [[0.1, 0.2]]

        chunks = [Chunk("a.py", 1, 1, "a", "a")]
        # No on_progress -- should not crash
        result = embed_chunks(chunks, mock_provider, task="document")
        assert len(result) == 1

    def test_progress_callback_called(self):
        """embed_chunks should call progress callback after each batch."""
        from nndex_cli.embed import embed_chunks

        mock_provider = MagicMock()
        mock_provider.get_embeddings.side_effect = [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.5, 0.6]],
        ]

        chunks = [
            Chunk("a.py", 1, 1, "a", "a"),
            Chunk("b.py", 1, 1, "b", "b"),
            Chunk("c.py", 1, 1, "c", "c"),
        ]

        progress_calls = []
        def on_progress(done, total):
            progress_calls.append((done, total))

        embed_chunks(chunks, mock_provider, task="document", batch_size=2,
                     on_progress=on_progress)

        # Should be called once per batch
        assert len(progress_calls) == 2
        assert progress_calls[0] == (2, 3)  # After first batch: 2 of 3 done
        assert progress_calls[1] == (3, 3)  # After second batch: 3 of 3 done


class TestHTTPEmbeddingProvider:
    """Test HTTP-based embedding provider for use without heylookitsanllm."""

    def test_get_embeddings_calls_api(self):
        """HTTPEmbeddingProvider should POST to /v1/embeddings."""
        from nndex_cli.http_provider import HTTPEmbeddingProvider

        provider = HTTPEmbeddingProvider(
            api_url="http://localhost:8000",
            model="embeddinggemma-300m",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }

        with patch("nndex_cli.http_provider.urllib.request.urlopen") as mock_urlopen:
            mock_urlopen.return_value.__enter__ = Mock(return_value=mock_response)
            mock_urlopen.return_value.__exit__ = Mock(return_value=False)
            # Mock the response read
            mock_response.read.return_value = json.dumps({
                "data": [
                    {"embedding": [0.1, 0.2, 0.3]},
                    {"embedding": [0.4, 0.5, 0.6]},
                ]
            }).encode()

            result = provider.get_embeddings(["hello", "world"], task="document")

        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]

    def test_has_get_embeddings_interface(self):
        """HTTPEmbeddingProvider should have the same interface as MLXEmbeddingProvider."""
        from nndex_cli.http_provider import HTTPEmbeddingProvider

        provider = HTTPEmbeddingProvider(
            api_url="http://localhost:8000",
            model="embeddinggemma-300m",
        )
        assert hasattr(provider, "get_embeddings")
        assert callable(provider.get_embeddings)


class TestLoadProvider:
    """Test provider loading logic."""

    def test_load_provider_prefers_mlx_when_available(self):
        """When heylookitsanllm is available, use MLXEmbeddingProvider."""
        from nndex_cli.main import _load_provider_auto

        mock_mlx_provider = MagicMock()
        with patch("nndex_cli.main._try_load_mlx_provider", return_value=mock_mlx_provider):
            provider = _load_provider_auto(model_path="/some/path")
        assert provider is mock_mlx_provider

    def test_load_provider_falls_back_to_http(self):
        """When heylookitsanllm is not available, fall back to HTTP."""
        from nndex_cli.main import _load_provider_auto

        with patch("nndex_cli.main._try_load_mlx_provider", return_value=None):
            provider = _load_provider_auto(
                model_path="/some/path",
                api_url="http://localhost:8000",
            )
        from nndex_cli.http_provider import HTTPEmbeddingProvider
        assert isinstance(provider, HTTPEmbeddingProvider)
