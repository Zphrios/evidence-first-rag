from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from uuid import UUID, uuid4

import httpx
import pytest
from openai import OpenAIError

from app.db.session import AsyncSessionLocal
from app.main import app
from app.models.document import Chunk, Document, DocumentPage, DocumentStatus, DocumentType
from app.models.project import Project
from app.services.embeddings import hash_text

EMBEDDING_DIMENSIONS = 1536
EMBEDDING_MODEL = "test-embedding-model"


def make_embedding() -> list[float]:
    """Create a deterministic vector compatible with the pgvector schema."""
    return [1.0] + [0.0] * (EMBEDDING_DIMENSIONS - 1)


class FakeOpenAIEmbeddingClient:
    """In-memory embedding provider used to keep API tests offline."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def create_embeddings(
        self,
        texts: Sequence[str],
        *,
        model: str,
    ) -> list[list[float]]:
        return [make_embedding() for _ in texts]


async def create_project_with_embedded_chunk() -> tuple[UUID, UUID]:
    """Create one processed project document with a retrievable chunk."""
    async with AsyncSessionLocal.begin() as session:
        project = Project(
            name=f"Evidence API Project {uuid4()}",
            description="Created by the evidence API integration test.",
        )
        session.add(project)
        await session.flush()

        document = Document(
            project_id=project.id,
            original_filename="general-conditions.pdf",
            storage_key=f"tests/evidence-api-{uuid4()}.pdf",
            sha256=sha256(str(uuid4()).encode("utf-8")).hexdigest(),
            content_type="application/pdf",
            size_bytes=1,
            document_type=DocumentType.CONSTRUCTION_CONTRACT,
            status=DocumentStatus.PROCESSED,
            processed_at=datetime.now(UTC),
        )
        session.add(document)
        await session.flush()

        chunk_text = (
            "The contractor shall provide fourteen days written notice before "
            "termination."
        )
        page = DocumentPage(
            document_id=document.id,
            page_number=9,
            text_content=chunk_text,
        )
        session.add(page)
        await session.flush()

        chunk = Chunk(
            page_id=page.id,
            chunk_index=1,
            text_content=chunk_text,
            embedding=make_embedding(),
            embedding_model=EMBEDDING_MODEL,
            embedding_text_hash=hash_text(chunk_text),
            embedded_at=datetime.now(UTC),
        )
        session.add(chunk)

        return project.id, document.id


async def delete_evidence_api_test_data(
    *,
    project_id: UUID | None,
    document_id: UUID | None,
) -> None:
    """Delete database rows created by evidence API integration tests."""
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
async def test_evidence_search_returns_404_when_project_is_missing() -> None:
    """A missing project is rejected before the embedding service is used."""
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            f"/projects/{uuid4()}/evidence-search",
            json={"question": "What notice is required before termination?"},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Project not found."}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"question": ""},
        {"question": " "},
        {"question": "\n\t"},
        {"question": "Valid question", "limit": 0},
        {"question": "Valid question", "limit": 21},
    ],
)
async def test_evidence_search_rejects_invalid_request_payload(
    payload: dict[str, str | int],
) -> None:
    """Invalid question text and limits produce FastAPI validation errors."""
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            f"/projects/{uuid4()}/evidence-search",
            json=payload,
        )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_evidence_search_returns_citation_ready_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The API returns page-citable evidence without calling OpenAI."""
    from app.api.routes import evidence

    monkeypatch.setattr(
        evidence,
        "OpenAIEmbeddingClient",
        FakeOpenAIEmbeddingClient,
    )
    monkeypatch.setattr(
        evidence,
        "get_settings",
        lambda: type(
            "TestSettings",
            (),
            {
                "openai_api_key": "test-api-key",
                "openai_embedding_model": EMBEDDING_MODEL,
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
            },
        )(),
    )

    project_id: UUID | None = None
    document_id: UUID | None = None

    try:
        project_id, document_id = await create_project_with_embedded_chunk()
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                f"/projects/{project_id}/evidence-search",
                json={
                    "question": "What written notice is required before termination?",
                    "limit": 4,
                },
            )

        assert response.status_code == 200

        payload = response.json()
        assert payload["question"] == (
            "What written notice is required before termination?"
        )
        assert payload["no_evidence_found"] is False
        assert len(payload["evidence"]) == 1

        citation = payload["evidence"][0]
        assert citation["document_id"] == str(document_id)
        assert citation["document_filename"] == "general-conditions.pdf"
        assert citation["page_number"] == 9
        assert citation["chunk_index"] == 1
        assert citation["excerpt"] == (
            "The contractor shall provide fourteen days written notice before "
            "termination."
        )
        assert UUID(citation["chunk_id"])
    finally:
        await delete_evidence_api_test_data(
            project_id=project_id,
            document_id=document_id,
        )


@pytest.mark.asyncio
async def test_evidence_search_returns_503_when_embedding_is_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A configured project fails safely when the embedding API key is absent."""
    from app.api.routes import evidence

    monkeypatch.setattr(
        evidence,
        "get_settings",
        lambda: type(
            "TestSettings",
            (),
            {
                "openai_api_key": None,
                "openai_embedding_model": EMBEDDING_MODEL,
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
            },
        )(),
    )

    project_id: UUID | None = None
    document_id: UUID | None = None

    try:
        project_id, document_id = await create_project_with_embedded_chunk()
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                f"/projects/{project_id}/evidence-search",
                json={"question": "What notice is required before termination?"},
            )

        assert response.status_code == 503
        assert response.json() == {
            "detail": "Embedding service is not configured.",
        }
    finally:
        await delete_evidence_api_test_data(
            project_id=project_id,
            document_id=document_id,
        )
    

@pytest.mark.asyncio
async def test_evidence_search_returns_503_when_embedding_provider_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provider failures return a safe service-unavailable response."""
    from app.api.routes import evidence

    class FailingOpenAIEmbeddingClient:
        def __init__(self, api_key: str) -> None:
            del api_key

        async def create_embeddings(
            self,
            texts: Sequence[str],
            *,
            model: str,
        ) -> list[list[float]]:
            del texts, model
            raise OpenAIError("Provider connection failed.")

    monkeypatch.setattr(
        evidence,
        "OpenAIEmbeddingClient",
        FailingOpenAIEmbeddingClient,
    )
    monkeypatch.setattr(
        evidence,
        "get_settings",
        lambda: type(
            "TestSettings",
            (),
            {
                "openai_api_key": "test-api-key",
                "openai_embedding_model": EMBEDDING_MODEL,
                "embedding_dimensions": EMBEDDING_DIMENSIONS,
            },
        )(),
    )

    project_id: UUID | None = None
    document_id: UUID | None = None

    try:
        project_id, document_id = await create_project_with_embedded_chunk()
        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            response = await client.post(
                f"/projects/{project_id}/evidence-search",
                json={"question": "What notice is required before termination?"},
            )

        assert response.status_code == 503
        assert response.json() == {
            "detail": "Unable to retrieve evidence at this time.",
        }
    finally:
        await delete_evidence_api_test_data(
            project_id=project_id,
            document_id=document_id,
        )