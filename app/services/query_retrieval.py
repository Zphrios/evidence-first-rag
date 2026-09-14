from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.evidence import Citation, EvidenceSearchResponse
from app.services.embeddings import EmbeddingClient, embed_texts
from app.services.retrieval import RetrievedChunk, retrieve_similar_chunks


async def retrieve_question_evidence(
    project_id: UUID,
    *,
    question: str,
    session: AsyncSession,
    client: EmbeddingClient,
    embedding_model: str,
    embedding_dimensions: int,
    limit: int = 8,
) -> list[RetrievedChunk]:
    """Embed one question and return project-scoped evidence chunks."""
    if not question.strip():
        raise ValueError("Question must not be blank.")

    embedded_questions = await embed_texts(
        [question],
        client=client,
        model=embedding_model,
        dimensions=embedding_dimensions,
    )

    return await retrieve_similar_chunks(
        project_id,
        query_embedding=embedded_questions[0].vector,
        session=session,
        embedding_model=embedding_model,
        limit=limit,
    )


async def search_project_evidence(
    project_id: UUID,
    *,
    question: str,
    session: AsyncSession,
    client: EmbeddingClient,
    embedding_model: str,
    embedding_dimensions: int,
    limit: int = 8,
) -> EvidenceSearchResponse:
    """Return citation-ready evidence for one project-scoped question."""
    retrieved_chunks = await retrieve_question_evidence(
        project_id,
        question=question,
        session=session,
        client=client,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_dimensions,
        limit=limit,
    )

    citations = [
        Citation(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_filename=chunk.document_filename,
            page_number=chunk.page_number,
            chunk_index=chunk.chunk_index,
            excerpt=chunk.text_content,
        )
        for chunk in retrieved_chunks
    ]

    return EvidenceSearchResponse(
        question=question,
        evidence=citations,
        no_evidence_found=not citations,
    )