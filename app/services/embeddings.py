from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from typing import Protocol


class EmbeddingClient(Protocol):
    """Provider contract for creating embeddings from text inputs."""

    async def create_embeddings(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> list[list[float]]:
        """Return one embedding vector for each input text."""


@dataclass(frozen=True)
class EmbeddedText:
    """Validated embedding output with provenance for one source text."""

    text_hash: str
    vector: list[float]
    model: str
    embedded_at: datetime


class EmbeddingValidationError(ValueError):
    """Raised when an embedding provider returns invalid output."""


def hash_text(text: str) -> str:
    """Return the SHA-256 digest for the exact text sent to the provider."""
    return sha256(text.encode("utf-8")).hexdigest()


async def embed_texts(
    texts: Sequence[str],
    *,
    client: EmbeddingClient,
    model: str,
    dimensions: int,
) -> list[EmbeddedText]:
    """Create validated embeddings without persisting source text or vectors."""
    if dimensions <= 0:
        raise ValueError("Embedding dimensions must be greater than zero.")

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Embedding text must not be blank.")

    vectors = await client.create_embeddings(texts, model=model)

    if len(vectors) != len(texts):
        raise EmbeddingValidationError(
            "Embedding provider returned a different number of vectors than texts."
        )

    embedded_at = datetime.now(UTC)
    embedded_texts: list[EmbeddedText] = []

    for text, vector in zip(texts, vectors, strict=True):
        if len(vector) != dimensions:
            raise EmbeddingValidationError(
                "Embedding provider returned a vector with an unexpected dimension."
            )

        embedded_texts.append(
            EmbeddedText(
                text_hash=hash_text(text),
                vector=vector,
                model=model,
                embedded_at=embedded_at,
            )
        )

    return embedded_texts