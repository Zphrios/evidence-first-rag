from dataclasses import dataclass
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Chunk, Document, DocumentPage, DocumentStatus
from app.services.embeddings import hash_text

STALE_CANDIDATE_MULTIPLIER = 3


@dataclass(frozen=True)
class RetrievedChunk:
    """Evidence chunk returned by a project-scoped semantic search."""

    chunk_id: UUID
    document_id: UUID
    document_filename: str
    page_number: int
    chunk_index: int
    text_content: str
    cosine_distance: float


async def retrieve_similar_chunks(
    project_id: UUID,
    *,
    query_embedding: list[float],
    session: AsyncSession,
    embedding_model: str,
    limit: int = 8,
) -> list[RetrievedChunk]:
    """Return nearest current chunks for one project using cosine distance."""
    if not query_embedding:
        raise ValueError("Query embedding must not be empty.")

    if limit <= 0:
        raise ValueError("Retrieval limit must be greater than zero.")

    candidate_limit = limit * STALE_CANDIDATE_MULTIPLIER
    
    distance = Chunk.embedding.cosine_distance(query_embedding).label(
        "cosine_distance"
    )

    result = await session.execute(
        select(
            Chunk.id,
            Document.id.label("document_id"),
            Document.original_filename,
            DocumentPage.page_number,
            Chunk.chunk_index,
            Chunk.text_content,
            Chunk.embedding_text_hash,
            distance,
        )
        .join(DocumentPage, Chunk.page_id == DocumentPage.id)
        .join(Document, DocumentPage.document_id == Document.id)
        .where(
            Document.project_id == project_id,
            Document.status == DocumentStatus.PROCESSED,
            Chunk.embedding.is_not(None),
            Chunk.embedding_model == embedding_model,
        )
        .order_by(distance)
        .limit(candidate_limit)
    )

    retrieved_chunks: list[RetrievedChunk] = []

    for row in result:
        if row.embedding_text_hash != hash_text(row.text_content):
            continue

        retrieved_chunks.append(
            RetrievedChunk(
                chunk_id=row.id,
                document_id=row.document_id,
                document_filename=row.original_filename,
                page_number=row.page_number,
                chunk_index=row.chunk_index,
                text_content=row.text_content,
                cosine_distance=float(row.cosine_distance),
            )
        )
        if len(retrieved_chunks) == limit:
            break

    return retrieved_chunks