"""HTTP embedding provider for OpenAI-compatible /v1/embeddings APIs.

Falls back to this when heylookitsanllm is not installed locally.
Works with any server that implements the OpenAI embeddings API.
"""

import json
import urllib.request
from typing import List, Optional


class HTTPEmbeddingProvider:
    """Embedding provider that calls a remote /v1/embeddings endpoint.

    Compatible with heylookitsanllm server, OpenAI, and other
    OpenAI-compatible embedding APIs.
    """

    def __init__(
        self,
        api_url: str = "http://localhost:8000",
        model: str = "embeddinggemma-300m",
    ):
        self.api_url = api_url.rstrip("/")
        self.model = model

    def get_embeddings(
        self,
        texts: List[str],
        task: Optional[str] = None,
        **kwargs,
    ) -> List[List[float]]:
        """Embed texts via HTTP API.

        Args:
            texts: List of strings to embed.
            task: Ignored for HTTP provider (server handles task prefixes).

        Returns:
            List of embedding vectors.
        """
        url = f"{self.api_url}/v1/embeddings"
        payload = json.dumps({
            "input": texts,
            "model": self.model,
        }).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read())

        return [item["embedding"] for item in body["data"]]
