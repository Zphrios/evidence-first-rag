from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID, uuid4

import pytest

from app.db.session import AsyncSessionLocal
from app.models.document import Chunk, Document, DocumentPage, DocumentStatus, DocumentType
from app.models.project import Project
from app.services.embeddings import hash_text
from app.services.query_retrieval import retrieve_question_evidence

EMBEDDING_DIMENSIONS = 1536
EMBEDDING_MODEL = "test-embedding-model"


def make_embedding(
    first_value: float,
    second_value: float = 0.0,
) -> list[float]:
    """Create a deterministic vector compatible with the pgvector schema."""
    embedding = [0.0] * EMBEDDING_DIMENSIONS
    embedding[0] = first_value
    embedding[1] = second_value
    return embedding


class FakeEmbeddingClient:
    """In-memory query embedding provider with no network access."""

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


async def create_project_with_chunk(
    *,
    project_name: str,
    text_content: str,
    embedding: list[float],
) -> tuple[UUID, UUID]:
    """Create one processed document containing one current embedded chunk."""
    async with AsyncSessionLocal.begin() as session:
        project = Project(
            name=project_name,
            description="Created by the query retrieval integration test.",
        )
        session.add(project)
        await session.flush()

        document = Document(
            project_id=project.id,
            original_filename="query-retrieval.pdf",
            storage_key=f"tests/query-retrieval-{uuid4()}.pdf",
            sha256=sha256(str(uuid4()).encode("utf-8")).hexdigest(),
            content_type="application/pdf",
            size_bytes=1,
            document_type=DocumentType.CONSTRUCTION_CONTRACT,
            status=DocumentStatus.PROCESSED,
            processed_at=datetime.now(UTC),
        )
        session.add(document)
        await session.flush()

        page = DocumentPage(
            document_id=document.id,
            page_number=4,
            text_content=text_content,
        )
        session.add(page)
        await session.flush()

        chunk = Chunk(
            page_id=page.id,
            chunk_index=2,
            text_content=text_content,
            embedding=embedding,
            embedding_model=EMBEDDING_MODEL,
            embedding_text_hash=hash_text(text_content),
            embedded_at=datetime.now(UTC),
        )
        session.add(chunk)

        return project.id, document.id


async def delete_query_retrieval_test_data(
    *,
    project_id: UUID | None,
    document_id: UUID | None,
) -> None:
    """Delete query retrieval test rows."""
    async with AsyncSessionLocal.begin() as session:
        if document_id is not None:
            document = await session.get(Document, document_id)
            if document is not None:
                await session.delete(document)

        if project_id is not None:
            project = await session.get(Project, project_id)
            if project is not None:
                await session.delete(project)


@pytest.mark.asyncio
async def test_retrieve_question_evidence_embeds_question_and_returns_evidence() -> None:
    """A question is embedded and returns page-cited evidence from its project."""
    project_id: UUID | None = None
    document_id: UUID | None = None

    question = "How much notice is required before termination?"
    evidence_text = "Either party must provide fourteen days written notice to terminate."

    client = FakeEmbeddingClient(vectors=[make_embedding(1.0)])

    try:
        project_id, document_id = await create_project_with_chunk(
            project_name=f"Question Retrieval Test {uuid4()}",
            text_content=evidence_text,
            embedding=make_embedding(1.0),
        )

        async with AsyncSessionLocal() as session:
            results = await retrieve_question_evidence(
                project_id,
                question=question,
                session=session,
                client=client,
                embedding_model=EMBEDDING_MODEL,
                embedding_dimensions=EMBEDDING_DIMENSIONS,
            )

        assert client.received_texts == [question]
        assert client.received_model == EMBEDDING_MODEL
        assert len(results) == 1
        assert results[0].document_id == document_id
        assert results[0].page_number == 4
        assert results[0].chunk_index == 2
        assert results[0].text_content == evidence_text
        assert results[0].cosine_distance == pytest.approx(0.0)
    finally:
        await delete_query_retrieval_test_data(
            project_id=project_id,
            document_id=document_id,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["", " ", "\n\t"])
async def test_retrieve_question_evidence_rejects_blank_question(
    question: str,
) -> None:
    """Blank questions fail before calling the embedding provider."""
    client = FakeEmbeddingClient(vectors=[make_embedding(1.0)])

    async with AsyncSessionLocal() as session:
        with pytest.raises(ValueError, match="Question must not be blank."):
            await retrieve_question_evidence(
                uuid4(),
                question=question,
                session=session,
                client=client,
                embedding_model=EMBEDDING_MODEL,
                embedding_dimensions=EMBEDDING_DIMENSIONS,
            )

    assert client.received_texts is None  
    

@pytest.mark.asyncio
async def test_retrieve_question_evidence_rejects_wrong_embedding_dimension() -> None:
    """A wrong-size query vector fails before semantic retrieval runs."""
    client = FakeEmbeddingClient(vectors=[[1.0, 0.0]])

    async with AsyncSessionLocal() as session:
        with pytest.raises(
            ValueError,
            match="unexpected dimension",
        ):
            await retrieve_question_evidence(
                uuid4(),
                question="How much notice is required before termination?",
                session=session,
                client=client,
                embedding_model=EMBEDDING_MODEL,
                embedding_dimensions=EMBEDDING_DIMENSIONS,
            )


@pytest.mark.asyncio
async def test_retrieve_question_evidence_propagates_provider_failure() -> None:
    """A provider error is not converted into empty or misleading evidence."""
    class FailingEmbeddingClient:
        async def create_embeddings(
            self,
            texts: Sequence[str],
            *,
            model: str,
        ) -> list[list[float]]:
            raise RuntimeError("Embedding provider is unavailable.")

    async with AsyncSessionLocal() as session:
        with pytest.raises(
            RuntimeError,
            match="Embedding provider is unavailable.",
        ):
            await retrieve_question_evidence(
                uuid4(),
                question="How much notice is required before termination?",
                session=session,
                client=FailingEmbeddingClient(),
                embedding_model=EMBEDDING_MODEL,
                embedding_dimensions=EMBEDDING_DIMENSIONS,
            )