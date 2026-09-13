from collections.abc import Sequence

import pytest

from app.services.embeddings import (
    EmbeddingValidationError,
    embed_texts,
    hash_text,
)


class FakeEmbeddingClient:
    """In-memory embedding provider used without network access."""

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.received_texts: Sequence[str] | None = None
        self.received_model: str | None = None

    async def create_embeddings(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> list[list[float]]:
        self.received_texts = texts
        self.received_model = model
        return self.vectors


@pytest.mark.asyncio
async def test_embed_texts_returns_validated_provenance() -> None:
    """Valid provider output returns vectors with hashes and model metadata."""
    texts = ["First chunk.", "Second chunk."]
    client = FakeEmbeddingClient(
        vectors=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    results = await embed_texts(
        texts,
        client=client,
        model="test-embedding-model",
        dimensions=3,
    )

    assert client.received_texts == texts
    assert client.received_model == "test-embedding-model"
    assert [result.text_hash for result in results] == [
        hash_text("First chunk."),
        hash_text("Second chunk."),
    ]
    assert [result.vector for result in results] == [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
    assert all(result.model == "test-embedding-model" for result in results)
    assert results[0].embedded_at == results[1].embedded_at
    assert results[0].embedded_at.tzinfo is not None


@pytest.mark.asyncio
async def test_embed_texts_returns_empty_list_without_provider_call() -> None:
    """An empty input does not call the embedding provider."""
    client = FakeEmbeddingClient(vectors=[])

    results = await embed_texts(
        [],
        client=client,
        model="test-embedding-model",
        dimensions=3,
    )

    assert results == []
    assert client.received_texts is None
    assert client.received_model is None


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["", " ", "\n\t"])
async def test_embed_texts_rejects_blank_text(text: str) -> None:
    """Blank chunk text is rejected before calling the provider."""
    client = FakeEmbeddingClient(vectors=[])

    with pytest.raises(ValueError, match="Embedding text must not be blank."):
        await embed_texts(
            [text],
            client=client,
            model="test-embedding-model",
            dimensions=3,
        )

    assert client.received_texts is None


@pytest.mark.asyncio
async def test_embed_texts_rejects_mismatched_vector_count() -> None:
    """Provider output must contain exactly one vector per input text."""
    client = FakeEmbeddingClient(vectors=[[1.0, 0.0, 0.0]])

    with pytest.raises(
        EmbeddingValidationError,
        match="different number of vectors than texts",
    ):
        await embed_texts(
            ["First chunk.", "Second chunk."],
            client=client,
            model="test-embedding-model",
            dimensions=3,
        )


@pytest.mark.asyncio
async def test_embed_texts_rejects_unexpected_vector_dimensions() -> None:
    """Every vector must match the configured database dimensions."""
    client = FakeEmbeddingClient(vectors=[[1.0, 0.0]])

    with pytest.raises(
        EmbeddingValidationError,
        match="unexpected dimension",
    ):
        await embed_texts(
            ["First chunk."],
            client=client,
            model="test-embedding-model",
            dimensions=3,
        )


@pytest.mark.asyncio
async def test_embed_texts_rejects_invalid_dimensions() -> None:
    """Configured dimensions must be positive."""
    client = FakeEmbeddingClient(vectors=[])

    with pytest.raises(
        ValueError,
        match="Embedding dimensions must be greater than zero.",
    ):
        await embed_texts(
            ["First chunk."],
            client=client,
            model="test-embedding-model",
            dimensions=0,
        )