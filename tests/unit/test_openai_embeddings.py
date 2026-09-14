from collections.abc import Sequence
from types import SimpleNamespace

import pytest

from app.services.openai_embeddings import OpenAIEmbeddingClient


class FakeEmbeddingsAPI:
    """Fake OpenAI embeddings API with no network access."""

    def __init__(self, response_data: list[SimpleNamespace]) -> None:
        self.response_data = response_data
        self.received_input: Sequence[str] | None = None
        self.received_model: str | None = None

    async def create(
        self,
        *,
        input: Sequence[str],
        model: str,
    ) -> SimpleNamespace:
        self.received_input = input
        self.received_model = model
        return SimpleNamespace(data=self.response_data)


class FailingEmbeddingsAPI:
    """Fake OpenAI embeddings API that raises a provider failure."""

    async def create(
        self,
        *,
        input: Sequence[str],
        model: str,
    ) -> SimpleNamespace:
        raise RuntimeError("OpenAI embeddings service unavailable.")


class FakeOpenAIClient:
    """Fake AsyncOpenAI client exposing the embeddings API."""

    def __init__(self, embeddings_api: FakeEmbeddingsAPI) -> None:
        self.embeddings = embeddings_api


@pytest.mark.asyncio
async def test_openai_embedding_client_returns_vectors() -> None:
    """The adapter maps OpenAI embedding response objects to vectors."""
    embeddings_api = FakeEmbeddingsAPI(
        response_data=[
            SimpleNamespace(embedding=[1.0, 0.0, 0.0]),
            SimpleNamespace(embedding=[0.0, 1.0, 0.0]),
        ]
    )
    client = OpenAIEmbeddingClient(
        "test-api-key",
        client=FakeOpenAIClient(embeddings_api),
    )

    vectors = await client.create_embeddings(
        ["First chunk.", "Second chunk."],
        model="test-embedding-model",
    )

    assert embeddings_api.received_input == ["First chunk.", "Second chunk."]
    assert embeddings_api.received_model == "test-embedding-model"
    assert vectors == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]


@pytest.mark.parametrize("api_key", ["", " ", "\n\t"])
def test_openai_embedding_client_rejects_blank_api_key(api_key: str) -> None:
    """The adapter rejects blank API keys before creating an SDK client."""
    with pytest.raises(ValueError, match="OpenAI API key must not be blank."):
        OpenAIEmbeddingClient(api_key)
        
        
@pytest.mark.asyncio
async def test_openai_embedding_client_propagates_provider_failure() -> None:
    """Provider failures are not converted into misleading embedding results."""
    client = OpenAIEmbeddingClient(
        "test-api-key",
        client=FakeOpenAIClient(FailingEmbeddingsAPI()),
    )

    with pytest.raises(
        RuntimeError,
        match="OpenAI embeddings service unavailable.",
    ):
        await client.create_embeddings(
            ["First chunk."],
            model="test-embedding-model",
        )