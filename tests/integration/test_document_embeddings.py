from collections.abc import Sequence
from hashlib import sha256
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.models.document import Chunk, Document, DocumentPage, DocumentStatus, DocumentType
from app.models.project import Project
from app.services.document_embeddings import embed_document_chunks
from app.services.embeddings import hash_text

EMBEDDING_DIMENSIONS = 1536

def make_embedding(value: float) -> list[float]:
    """Create a deterministic vector compatible with the pgvector schema."""
    embedding = [0.0] * EMBEDDING_DIMENSIONS
    embedding[0] = value
    return embedding


class FakeEmbeddingClient:
    """In-memory provider used without network access."""

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


async def create_processed_document_with_chunks(
    *,
    text_contents: list[str],
) -> tuple[UUID, UUID, list[UUID]]:
    """Create one processed document with page-aware chunks for testing."""
    async with AsyncSessionLocal.begin() as session:
        project = Project(
            name=f"Embedding Workflow Test {uuid4()}",
            description="Created by the document embeddings integration test.",
        )
        session.add(project)
        await session.flush()

        document = Document(
            project_id=project.id,
            original_filename="embedding-workflow.pdf",
            storage_key=f"tests/embedding-workflow-{uuid4()}.pdf",
            sha256=sha256(str(uuid4()).encode("utf-8")).hexdigest(),
            content_type="application/pdf",
            size_bytes=1,
            document_type=DocumentType.CONSTRUCTION_CONTRACT,
            status=DocumentStatus.PROCESSED,
        )
        session.add(document)
        await session.flush()

        chunk_ids: list[UUID] = []

        for page_number, text_content in enumerate(text_contents, start=1):
            page = DocumentPage(
                document_id=document.id,
                page_number=page_number,
                text_content=text_content,
            )
            session.add(page)
            await session.flush()

            chunk = Chunk(
                page_id=page.id,
                chunk_index=0,
                text_content=text_content,
            )
            session.add(chunk)
            await session.flush()
            chunk_ids.append(chunk.id)

        return project.id, document.id, chunk_ids


async def delete_embedding_test_data(
    *,
    project_id: UUID | None,
    document_id: UUID | None,
) -> None:
    """Delete test rows in dependency order."""
    async with AsyncSessionLocal.begin() as session:
        if document_id is not None:
            chunk_result = await session.execute(
                select(Chunk)
                .join(DocumentPage)
                .where(DocumentPage.document_id == document_id)
            )
            for chunk in chunk_result.scalars():
                await session.delete(chunk)

            page_result = await session.execute(
                select(DocumentPage).where(DocumentPage.document_id == document_id)
            )
            for page in page_result.scalars():
                await session.delete(page)

            document = await session.get(Document, document_id)
            if document is not None:
                await session.delete(document)

        if project_id is not None:
            project = await session.get(Project, project_id)
            if project is not None:
                await session.delete(project)


@pytest.mark.asyncio
async def test_embed_document_chunks_persists_validated_embeddings() -> None:
    """The workflow embeds all stale chunks and persists their provenance."""
    project_id: UUID | None = None
    document_id: UUID | None = None

    first_text = "The contractor shall provide written notice before termination."
    second_text = "Payment is due within thirty days after approval."
    client = FakeEmbeddingClient(
        vectors=[
            make_embedding(1.0),
            make_embedding(2.0),
        ]
    )

    try:
        project_id, document_id, chunk_ids = await create_processed_document_with_chunks(
            text_contents=[first_text, second_text],
        )

        async with AsyncSessionLocal.begin() as session:
            embedded_count = await embed_document_chunks(
                document_id,
                session=session,
                client=client,
                model="test-embedding-model",
                dimensions=EMBEDDING_DIMENSIONS,
            )

            assert embedded_count == 2

        assert client.received_texts == [first_text, second_text]
        assert client.received_model == "test-embedding-model"

        async with AsyncSessionLocal() as session:
            stored_chunks_result = await session.execute(
                select(Chunk)
                .join(DocumentPage)
                .where(Chunk.id.in_(chunk_ids))
                .order_by(DocumentPage.page_number, Chunk.chunk_index)
            )
            stored_chunks = list(stored_chunks_result.scalars().all())

            assert len(stored_chunks) == 2
            assert [chunk.embedding for chunk in stored_chunks] == [
                make_embedding(1.0),
                make_embedding(2.0),
            ]
            assert [chunk.embedding_text_hash for chunk in stored_chunks] == [
                hash_text(first_text),
                hash_text(second_text),
            ]
            assert all(chunk.embedded_at is not None for chunk in stored_chunks)
    finally:
        await delete_embedding_test_data(
            project_id=project_id,
            document_id=document_id,
        )


@pytest.mark.asyncio
async def test_embed_document_chunks_skips_current_embeddings() -> None:
    """The workflow skips chunks whose vector, model, and text hash are current."""
    project_id: UUID | None = None
    document_id: UUID | None = None

    text_content = "The contractor must maintain required insurance coverage."
    client = FakeEmbeddingClient(vectors=[make_embedding(1.0)])

    try:
        project_id, document_id, _ = await create_processed_document_with_chunks(
            text_contents=[text_content],
        )

        async with AsyncSessionLocal.begin() as session:
            first_count = await embed_document_chunks(
                document_id,
                session=session,
                client=client,
                model="test-embedding-model",
                dimensions=EMBEDDING_DIMENSIONS,
            )
            assert first_count == 1

        client.vectors = []

        async with AsyncSessionLocal.begin() as session:
            second_count = await embed_document_chunks(
                document_id,
                session=session,
                client=client,
                model="test-embedding-model",
                dimensions=EMBEDDING_DIMENSIONS,
            )

            assert second_count == 0

        assert client.received_texts == [text_content]
    finally:
        await delete_embedding_test_data(
            project_id=project_id,
            document_id=document_id,
        )


@pytest.mark.asyncio
async def test_embed_document_chunks_rolls_back_on_invalid_provider_output() -> None:
    """Invalid provider output leaves every chunk without a partial embedding."""
    project_id: UUID | None = None
    document_id: UUID | None = None

    first_text = "First contract clause."
    second_text = "Second contract clause."
    client = FakeEmbeddingClient(
        vectors=[
            make_embedding(1.0),
        ]
    )

    try:
        project_id, document_id, chunk_ids = await create_processed_document_with_chunks(
            text_contents=[first_text, second_text],
        )

        with pytest.raises(
            ValueError,
            match="different number of vectors than texts",
        ):
            async with AsyncSessionLocal.begin() as session:
                await embed_document_chunks(
                    document_id,
                    session=session,
                    client=client,
                    model="test-embedding-model",
                    dimensions=EMBEDDING_DIMENSIONS,
                )

        async with AsyncSessionLocal() as session:
            stored_chunks_result = await session.execute(
                select(Chunk)
                .join(DocumentPage)
                .where(Chunk.id.in_(chunk_ids))
                .order_by(DocumentPage.page_number, Chunk.chunk_index)
            )
            stored_chunks = list(stored_chunks_result.scalars().all())

            assert len(stored_chunks) == 2
            assert all(chunk.embedding is None for chunk in stored_chunks)
            assert all(chunk.embedding_model is None for chunk in stored_chunks)
            assert all(chunk.embedding_text_hash is None for chunk in stored_chunks)
            assert all(chunk.embedded_at is None for chunk in stored_chunks)
    finally:
        await delete_embedding_test_data(
            project_id=project_id,
            document_id=document_id,
        )