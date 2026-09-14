from collections.abc import Sequence
from typing import Protocol

from openai import AsyncOpenAI


class OpenAIEmbeddingsAPI(Protocol):
    """Subset of the OpenAI embeddings API used by this adapter."""

    async def create(
        self,
        *,
        input: Sequence[str],
        model: str,
    ) -> object:
        """Create embedding vectors for the supplied text inputs."""


class OpenAIAsyncClient(Protocol):
    """Subset of AsyncOpenAI used by the embedding adapter."""

    embeddings: OpenAIEmbeddingsAPI


class OpenAIEmbeddingClient:
    """Embedding provider adapter backed by the OpenAI async SDK."""

    def __init__(
        self,
        api_key: str,
        *,
        client: OpenAIAsyncClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key must not be blank.")

        self._client = client or AsyncOpenAI(api_key=api_key)

    async def create_embeddings(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> list[list[float]]:
        """Return vectors from OpenAI without logging source texts or embeddings."""
        response = await self._client.embeddings.create(
            input=texts,
            model=model,
        )

        return [item.embedding for item in response.data]