from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Chunk, DocumentPage
from app.services.embeddings import EmbeddingClient, embed_texts, hash_text


async def embed_document_chunks(
    document_id: UUID,
    *,
    session: AsyncSession,
    client: EmbeddingClient,
    model: str,
    dimensions: int,
) -> int:
    """Create and atomically persist embeddings for stale chunks in one document."""
    result = await session.execute(
        select(Chunk)
        .join(DocumentPage)
        .where(DocumentPage.document_id == document_id)
        .order_by(DocumentPage.page_number, Chunk.chunk_index)
        .with_for_update()
    )
    chunks = list(result.scalars().all())

    chunks_to_embed = [
        chunk
        for chunk in chunks
        if chunk.embedding is None
        or chunk.embedding_model != model
        or chunk.embedding_text_hash != hash_text(chunk.text_content)
    ]

    if not chunks_to_embed:
        return 0

    embedded_texts = await embed_texts(
        [chunk.text_content for chunk in chunks_to_embed],
        client=client,
        model=model,
        dimensions=dimensions,
    )

    for chunk, embedded_text in zip(
        chunks_to_embed,
        embedded_texts,
        strict=True,
    ):
        chunk.embedding = embedded_text.vector
        chunk.embedding_model = embedded_text.model
        chunk.embedding_text_hash = embedded_text.text_hash
        chunk.embedded_at = embedded_text.embedded_at

    return len(chunks_to_embed)