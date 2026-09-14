from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID, uuid4

import pytest

from app.db.session import AsyncSessionLocal
from app.models.document import Chunk, Document, DocumentPage, DocumentStatus, DocumentType
from app.models.project import Project
from app.services.embeddings import hash_text
from app.services.retrieval import retrieve_similar_chunks

EMBEDDING_DIMENSIONS = 1536
EMBEDDING_MODEL = "test-embedding-model"


def make_embedding(value: float) -> list[float]:
    """Create a deterministic embedding compatible with the pgvector schema."""
    embedding = [0.0] * EMBEDDING_DIMENSIONS
    embedding[0] = value
    return embedding


def make_ranked_embedding(
    first_value: float,
    second_value: float,
) -> list[float]:
    """Create a deterministic vector with a distinct cosine-distance ranking."""
    embedding = [0.0] * EMBEDDING_DIMENSIONS
    embedding[0] = first_value
    embedding[1] = second_value
    return embedding


async def create_project_with_embedded_chunk(
    *,
    project_name: str,
    text_content: str,
    embedding: list[float] | None,
    embedding_model: str | None = EMBEDDING_MODEL,
    embedding_text_hash: str | None = None,
    document_status: DocumentStatus = DocumentStatus.PROCESSED,
) -> tuple[UUID, UUID]:
    """Create one project, document, page, and chunk for retrieval testing."""
    async with AsyncSessionLocal.begin() as session:
        project = Project(
            name=project_name,
            description="Created by the retrieval integration test.",
        )
        session.add(project)
        await session.flush()

        document = Document(
            project_id=project.id,
            original_filename="retrieval-test.pdf",
            storage_key=f"tests/retrieval-{uuid4()}.pdf",
            sha256=sha256(str(uuid4()).encode("utf-8")).hexdigest(),
            content_type="application/pdf",
            size_bytes=1,
            document_type=DocumentType.CONSTRUCTION_CONTRACT,
            status=document_status,
            processed_at=datetime.now(UTC),
        )
        session.add(document)
        await session.flush()

        page = DocumentPage(
            document_id=document.id,
            page_number=1,
            text_content=text_content,
        )
        session.add(page)
        await session.flush()

        chunk = Chunk(
            page_id=page.id,
            chunk_index=0,
            text_content=text_content,
            embedding=embedding,
            embedding_model=embedding_model,
            embedding_text_hash=embedding_text_hash or hash_text(text_content),
            embedded_at=datetime.now(UTC) if embedding is not None else None,
        )
        session.add(chunk)

        return project.id, document.id


async def delete_retrieval_test_data(
    *,
    project_ids: list[UUID],
    document_ids: list[UUID],
) -> None:
    """Delete retrieval test rows in dependency order."""
    async with AsyncSessionLocal.begin() as session:
        for document_id in document_ids:
            document = await session.get(Document, document_id)
            if document is not None:
                await session.delete(document)

        for project_id in project_ids:
            project = await session.get(Project, project_id)
            if project is not None:
                await session.delete(project)


@pytest.mark.asyncio
async def test_retrieval_returns_ranked_chunks_from_requested_project_only() -> None:
    """Semantic retrieval ranks valid chunks and never leaks another project."""
    requested_project_id: UUID | None = None
    requested_document_id: UUID | None = None
    other_project_id: UUID | None = None
    other_document_id: UUID | None = None

    requested_text = "Termination requires fourteen days written notice."
    other_project_text = "A different project contains unrelated payment terms."

    try:
        (
            requested_project_id,
            requested_document_id,
        ) = await create_project_with_embedded_chunk(
            project_name=f"Requested Retrieval Project {uuid4()}",
            text_content=requested_text,
            embedding=make_embedding(1.0),
        )
        (
            other_project_id,
            other_document_id,
        ) = await create_project_with_embedded_chunk(
            project_name=f"Other Retrieval Project {uuid4()}",
            text_content=other_project_text,
            embedding=make_embedding(0.99),
        )

        async with AsyncSessionLocal() as session:
            results = await retrieve_similar_chunks(
                requested_project_id,
                query_embedding=make_embedding(1.0),
                session=session,
                embedding_model=EMBEDDING_MODEL,
            )

        assert len(results) == 1
        assert results[0].document_id == requested_document_id
        assert results[0].text_content == requested_text
        assert results[0].page_number == 1
        assert results[0].chunk_index == 0
        assert results[0].cosine_distance == pytest.approx(0.0)
    finally:
        await delete_retrieval_test_data(
            project_ids=[
                project_id
                for project_id in [requested_project_id, other_project_id]
                if project_id is not None
            ],
            document_ids=[
                document_id
                for document_id in [requested_document_id, other_document_id]
                if document_id is not None
            ],
        )


@pytest.mark.asyncio
async def test_retrieval_excludes_stale_or_unprocessed_chunks() -> None:
    """Only current embeddings from processed documents qualify as evidence."""
    project_id: UUID | None = None
    current_document_id: UUID | None = None
    stale_document_id: UUID | None = None
    unprocessed_document_id: UUID | None = None

    current_text = "The current clause is eligible for retrieval."
    stale_text = "This text changed after its embedding was created."
    unprocessed_text = "This document is uploaded but not processed."

    try:
        project_id, current_document_id = await create_project_with_embedded_chunk(
            project_name=f"Retrieval State Test {uuid4()}",
            text_content=current_text,
            embedding=make_embedding(1.0),
        )

        async with AsyncSessionLocal.begin() as session:
            project = await session.get(Project, project_id)
            assert project is not None

            stale_document = Document(
                project_id=project.id,
                original_filename="stale-retrieval-test.pdf",
                storage_key=f"tests/stale-retrieval-{uuid4()}.pdf",
                sha256=sha256(str(uuid4()).encode("utf-8")).hexdigest(),
                content_type="application/pdf",
                size_bytes=1,
                document_type=DocumentType.CONSTRUCTION_CONTRACT,
                status=DocumentStatus.PROCESSED,
                processed_at=datetime.now(UTC),
            )
            session.add(stale_document)
            await session.flush()
            stale_document_id = stale_document.id

            stale_page = DocumentPage(
                document_id=stale_document.id,
                page_number=1,
                text_content=stale_text,
            )
            session.add(stale_page)
            await session.flush()

            stale_chunk = Chunk(
                page_id=stale_page.id,
                chunk_index=0,
                text_content=stale_text,
                embedding=make_embedding(0.99),
                embedding_model=EMBEDDING_MODEL,
                embedding_text_hash=hash_text("Previous clause text."),
                embedded_at=datetime.now(UTC),
            )
            session.add(stale_chunk)

            unprocessed_document = Document(
                project_id=project.id,
                original_filename="unprocessed-retrieval-test.pdf",
                storage_key=f"tests/unprocessed-retrieval-{uuid4()}.pdf",
                sha256=sha256(str(uuid4()).encode("utf-8")).hexdigest(),
                content_type="application/pdf",
                size_bytes=1,
                document_type=DocumentType.CONSTRUCTION_CONTRACT,
                status=DocumentStatus.UPLOADED,
            )
            session.add(unprocessed_document)
            await session.flush()
            unprocessed_document_id = unprocessed_document.id

            unprocessed_page = DocumentPage(
                document_id=unprocessed_document.id,
                page_number=1,
                text_content=unprocessed_text,
            )
            session.add(unprocessed_page)
            await session.flush()

            unprocessed_chunk = Chunk(
                page_id=unprocessed_page.id,
                chunk_index=0,
                text_content=unprocessed_text,
                embedding=make_embedding(0.98),
                embedding_model=EMBEDDING_MODEL,
                embedding_text_hash=hash_text(unprocessed_text),
                embedded_at=datetime.now(UTC),
            )
            session.add(unprocessed_chunk)

        async with AsyncSessionLocal() as session:
            results = await retrieve_similar_chunks(
                project_id,
                query_embedding=make_embedding(1.0),
                session=session,
                embedding_model=EMBEDDING_MODEL,
            )

        assert len(results) == 1
        assert results[0].document_id == current_document_id
        assert results[0].text_content == current_text
    finally:
        await delete_retrieval_test_data(
            project_ids=[project_id] if project_id is not None else [],
            document_ids=[
                document_id
                for document_id in [
                    current_document_id,
                    stale_document_id,
                    unprocessed_document_id,
                ]
                if document_id is not None
            ],
        )


@pytest.mark.asyncio
async def test_retrieval_rejects_invalid_query_or_limit() -> None:
    """Invalid retrieval parameters fail before querying the database."""
    async with AsyncSessionLocal() as session:
        with pytest.raises(ValueError, match="Query embedding must not be empty."):
            await retrieve_similar_chunks(
                uuid4(),
                query_embedding=[],
                session=session,
                embedding_model=EMBEDDING_MODEL,
            )

        with pytest.raises(ValueError, match="Retrieval limit must be greater than zero."):
            await retrieve_similar_chunks(
                uuid4(),
                query_embedding=make_embedding(1.0),
                session=session,
                embedding_model=EMBEDDING_MODEL,
                limit=0,
            )
            

@pytest.mark.asyncio
async def test_retrieval_over_fetches_when_nearest_chunks_are_stale() -> None:
    """Freshness filtering still returns requested valid results after stale candidates."""
    project_id: UUID | None = None
    document_ids: list[UUID] = []

    stale_first_text = "First stale candidate."
    stale_second_text = "Second stale candidate."
    valid_first_text = "First valid contract clause."
    valid_second_text = "Second valid contract clause."

    try:
        project_id, stale_first_document_id = (
            await create_project_with_embedded_chunk(
                project_name=f"Over Fetch Retrieval Test {uuid4()}",
                text_content=stale_first_text,
                embedding=make_ranked_embedding(1.0, 0.0),
                embedding_text_hash=hash_text("Previous first candidate."),
            )
        )
        document_ids.append(stale_first_document_id)

        async with AsyncSessionLocal.begin() as session:
            project = await session.get(Project, project_id)
            assert project is not None

            for text_content, embedding, stored_hash in [
                (
                    stale_second_text,
                    make_ranked_embedding(0.99, 0.1),
                    hash_text("Previous second candidate."),
                ),
                (
                    valid_first_text,
                    make_ranked_embedding(0.98, 0.2),
                    hash_text(valid_first_text),
                ),
                (
                    valid_second_text,
                    make_ranked_embedding(0.97, 0.3),
                    hash_text(valid_second_text),
                ),
            ]:
                document = Document(
                    project_id=project.id,
                    original_filename="over-fetch-retrieval.pdf",
                    storage_key=f"tests/over-fetch-{uuid4()}.pdf",
                    sha256=sha256(str(uuid4()).encode("utf-8")).hexdigest(),
                    content_type="application/pdf",
                    size_bytes=1,
                    document_type=DocumentType.CONSTRUCTION_CONTRACT,
                    status=DocumentStatus.PROCESSED,
                    processed_at=datetime.now(UTC),
                )
                session.add(document)
                await session.flush()
                document_ids.append(document.id)

                page = DocumentPage(
                    document_id=document.id,
                    page_number=1,
                    text_content=text_content,
                )
                session.add(page)
                await session.flush()

                chunk = Chunk(
                    page_id=page.id,
                    chunk_index=0,
                    text_content=text_content,
                    embedding=embedding,
                    embedding_model=EMBEDDING_MODEL,
                    embedding_text_hash=stored_hash,
                    embedded_at=datetime.now(UTC),
                )
                session.add(chunk)

        async with AsyncSessionLocal() as session:
            results = await retrieve_similar_chunks(
                project_id,
                query_embedding=make_ranked_embedding(1.0, 0.0),
                session=session,
                embedding_model=EMBEDDING_MODEL,
                limit=2,
            )

        assert [result.text_content for result in results] == [
            valid_first_text,
            valid_second_text,
        ]
    finally:
        await delete_retrieval_test_data(
            project_ids=[project_id] if project_id is not None else [],
            document_ids=document_ids,
        )