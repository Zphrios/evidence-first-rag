from datetime import UTC, datetime
from hashlib import sha256
from types import SimpleNamespace
from uuid import UUID, uuid4

import httpx
import pymupdf
import pytest
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.main import app
from app.models.audit import AuditAction, AuditEvent
from app.models.document import (
    Chunk,
    Document,
    DocumentPage,
    DocumentStatus,
    DocumentType,
)
from app.models.project import Project
from app.services.storage import STORAGE_ROOT


def build_text_pdf() -> bytes:
    """Create a valid two-page PDF with an extractable text layer."""
    pdf_document = pymupdf.open()

    first_page = pdf_document.new_page()
    first_page.insert_text(
        (72, 72),
        "Contract page one. The contractor shall submit a payment application.",
    )

    second_page = pdf_document.new_page()
    second_page.insert_text(
        (72, 72),
        "Contract page two. Payment is due within thirty days after approval.",
    )

    pdf_bytes = pdf_document.tobytes()
    pdf_document.close()
    return pdf_bytes

def build_three_page_text_pdf() -> bytes:
    """Create a valid three-page PDF with extractable text on every page."""
    pdf_document = pymupdf.open()

    for page_number in range(1, 4):
        page = pdf_document.new_page()
        page.insert_text(
            (72, 72),
            f"Page {page_number} contains extractable contract text.",
        )

    pdf_bytes = pdf_document.tobytes()
    pdf_document.close()
    return pdf_bytes

@pytest.mark.asyncio
async def test_process_pdf_persists_pages_chunks_and_audit_events() -> None:
    """Processing a text PDF saves page-cited chunks and audit evidence."""
    trace_id = f"test-processing-{uuid4()}"
    project_name = f"PDF Processing Test {uuid4()}"
    pdf_bytes = build_text_pdf()

    transport = httpx.ASGITransport(app=app)
    project_id: UUID | None = None
    document_id: UUID | None = None
    storage_key: str | None = None

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            project_response = await client.post(
                "/projects",
                headers={"X-Trace-Id": trace_id},
                json={
                    "name": project_name,
                    "description": "Created by the PDF processing integration test.",
                },
            )
            assert project_response.status_code == 201
            project_id = UUID(project_response.json()["id"])

            upload_response = await client.post(
                f"/projects/{project_id}/documents",
                headers={"X-Trace-Id": trace_id},
                data={"document_type": "construction_contract"},
                files={
                    "file": (
                        "processing-contract.pdf",
                        pdf_bytes,
                        "application/pdf",
                    )
                },
            )
            assert upload_response.status_code == 201

            uploaded_document = upload_response.json()
            document_id = UUID(uploaded_document["id"])
            storage_key = uploaded_document["storage_key"]

            process_response = await client.post(
                f"/documents/{document_id}/process",
                headers={"X-Trace-Id": trace_id},
            )

            assert process_response.status_code == 200
            assert process_response.headers["X-Trace-Id"] == trace_id

            processing_result = process_response.json()
            assert processing_result["document"]["id"] == str(document_id)
            assert processing_result["document"]["status"] == "processed"
            assert processing_result["document"]["processed_at"] is not None
            assert processing_result["document"]["error_message"] is None
            assert processing_result["page_count"] == 2
            assert processing_result["chunk_count"] == 2

            duplicate_process_response = await client.post(
                f"/documents/{document_id}/process",
                headers={"X-Trace-Id": f"{trace_id}-duplicate"},
            )

            assert duplicate_process_response.status_code == 409
            assert duplicate_process_response.json()["detail"] == (
                "Document has already been processed."
            )

        async with AsyncSessionLocal() as session:
            document = await session.get(Document, document_id)
            assert document is not None
            assert document.status == DocumentStatus.PROCESSED
            assert document.processed_at is not None
            assert document.error_message is None

            page_result = await session.execute(
                select(DocumentPage)
                .where(DocumentPage.document_id == document_id)
                .order_by(DocumentPage.page_number)
            )
            pages = list(page_result.scalars().all())

            assert [page.page_number for page in pages] == [1, 2]
            assert "Contract page one." in pages[0].text_content
            assert "Contract page two." in pages[1].text_content

            chunk_result = await session.execute(
                select(Chunk)
                .join(DocumentPage)
                .where(DocumentPage.document_id == document_id)
                .order_by(DocumentPage.page_number, Chunk.chunk_index)
            )
            chunks = list(chunk_result.scalars().all())

            assert len(chunks) == 2
            assert [chunk.chunk_index for chunk in chunks] == [0, 0]
            assert "payment application" in chunks[0].text_content
            assert "thirty days" in chunks[1].text_content

            started_result = await session.execute(
                select(AuditEvent).where(
                    AuditEvent.document_id == document_id,
                    AuditEvent.action == AuditAction.DOCUMENT_PROCESSING_STARTED,
                    AuditEvent.trace_id == trace_id,
                )
            )
            started_event = started_result.scalar_one()

            assert started_event.project_id == project_id
            assert started_event.event_data == {"storage_key": storage_key}

            processed_result = await session.execute(
                select(AuditEvent).where(
                    AuditEvent.document_id == document_id,
                    AuditEvent.action == AuditAction.DOCUMENT_PROCESSED,
                    AuditEvent.trace_id == trace_id,
                )
            )
            processed_event = processed_result.scalar_one()

            assert processed_event.project_id == project_id
            assert processed_event.event_data == {
                "page_count": 2,
                "chunk_count": 2,
            }
    finally:
        if storage_key is not None:
            storage_path = STORAGE_ROOT / storage_key
            if storage_path.exists():
                storage_path.unlink()

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

                audit_result = await session.execute(
                    select(AuditEvent).where(AuditEvent.document_id == document_id)
                )
                for audit_event in audit_result.scalars():
                    await session.delete(audit_event)

                document = await session.get(Document, document_id)
                if document is not None:
                    await session.delete(document)

            if project_id is not None:
                audit_result = await session.execute(
                    select(AuditEvent).where(AuditEvent.project_id == project_id)
                )
                for audit_event in audit_result.scalars():
                    await session.delete(audit_event)

                project = await session.get(Project, project_id)
                if project is not None:
                    await session.delete(project)


@pytest.mark.asyncio
async def test_process_blank_pdf_marks_document_failed_and_records_audit_event() -> None:
    """A PDF without extractable text fails cleanly and preserves audit evidence."""
    trace_id = f"test-processing-failure-{uuid4()}"
    project_name = f"Blank PDF Processing Test {uuid4()}"

    pdf_document = pymupdf.open()
    pdf_document.new_page()
    blank_pdf_bytes = pdf_document.tobytes()
    pdf_document.close()

    transport = httpx.ASGITransport(app=app)
    project_id: UUID | None = None
    document_id: UUID | None = None
    storage_key: str | None = None

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            project_response = await client.post(
                "/projects",
                headers={"X-Trace-Id": trace_id},
                json={
                    "name": project_name,
                    "description": "Created by the blank PDF processing integration test.",
                },
            )
            assert project_response.status_code == 201
            project_id = UUID(project_response.json()["id"])

            upload_response = await client.post(
                f"/projects/{project_id}/documents",
                headers={"X-Trace-Id": trace_id},
                data={"document_type": "construction_contract"},
                files={
                    "file": (
                        "blank-processing-contract.pdf",
                        blank_pdf_bytes,
                        "application/pdf",
                    )
                },
            )
            assert upload_response.status_code == 201

            uploaded_document = upload_response.json()
            document_id = UUID(uploaded_document["id"])
            storage_key = uploaded_document["storage_key"]

            process_response = await client.post(
                f"/documents/{document_id}/process",
                headers={"X-Trace-Id": trace_id},
            )

            assert process_response.status_code == 422
            assert process_response.headers["X-Trace-Id"] == trace_id
            assert process_response.json()["detail"] == (
                "PDF contains no extractable text layer "
                "(scanned/raster document requires OCR)."
            )

        async with AsyncSessionLocal() as session:
            document = await session.get(Document, document_id)
            assert document is not None
            assert document.status == DocumentStatus.FAILED
            assert document.processed_at is None
            assert document.error_message == (
                "PDF contains no extractable text layer "
                "(scanned/raster document requires OCR)."
            )

            page_result = await session.execute(
                select(DocumentPage).where(DocumentPage.document_id == document_id)
            )
            assert list(page_result.scalars().all()) == []

            chunk_result = await session.execute(
                select(Chunk)
                .join(DocumentPage)
                .where(DocumentPage.document_id == document_id)
            )
            assert list(chunk_result.scalars().all()) == []

            started_result = await session.execute(
                select(AuditEvent).where(
                    AuditEvent.document_id == document_id,
                    AuditEvent.action == AuditAction.DOCUMENT_PROCESSING_STARTED,
                    AuditEvent.trace_id == trace_id,
                )
            )
            started_event = started_result.scalar_one()
            assert started_event.event_data == {"storage_key": storage_key}

            failed_result = await session.execute(
                select(AuditEvent).where(
                    AuditEvent.document_id == document_id,
                    AuditEvent.action == AuditAction.DOCUMENT_FAILED,
                    AuditEvent.trace_id == trace_id,
                )
            )
            failed_event = failed_result.scalar_one()
            assert failed_event.project_id == project_id
            assert failed_event.event_data == {
                "reason": (
                    "PDF contains no extractable text layer "
                    "(scanned/raster document requires OCR)."
                )
            }
    finally:
        if storage_key is not None:
            storage_path = STORAGE_ROOT / storage_key
            if storage_path.exists():
                storage_path.unlink()

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

                audit_result = await session.execute(
                    select(AuditEvent).where(AuditEvent.document_id == document_id)
                )
                for audit_event in audit_result.scalars():
                    await session.delete(audit_event)

                document = await session.get(Document, document_id)
                if document is not None:
                    await session.delete(document)

            if project_id is not None:
                audit_result = await session.execute(
                    select(AuditEvent).where(AuditEvent.project_id == project_id)
                )
                for audit_event in audit_result.scalars():
                    await session.delete(audit_event)

                project = await session.get(Project, project_id)
                if project is not None:
                    await session.delete(project)
  
                    
@pytest.mark.asyncio
async def test_get_document_returns_metadata_and_processing_status() -> None:
    """Fetching a document returns metadata and its current processing status."""
    trace_id = f"test-document-status-{uuid4()}"
    project_name = f"Document Status Test {uuid4()}"
    pdf_bytes = build_text_pdf()

    transport = httpx.ASGITransport(app=app)
    project_id: UUID | None = None
    document_id: UUID | None = None
    storage_key: str | None = None

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            project_response = await client.post(
                "/projects",
                headers={"X-Trace-Id": trace_id},
                json={
                    "name": project_name,
                    "description": "Created by the document status integration test.",
                },
            )
            assert project_response.status_code == 201
            project_id = UUID(project_response.json()["id"])

            upload_response = await client.post(
                f"/projects/{project_id}/documents",
                headers={"X-Trace-Id": trace_id},
                data={"document_type": "construction_contract"},
                files={
                    "file": (
                        "document-status.pdf",
                        pdf_bytes,
                        "application/pdf",
                    )
                },
            )
            assert upload_response.status_code == 201

            uploaded_document = upload_response.json()
            document_id = UUID(uploaded_document["id"])
            storage_key = uploaded_document["storage_key"]

            get_response = await client.get(
                f"/documents/{document_id}",
                headers={"X-Trace-Id": trace_id},
            )

            assert get_response.status_code == 200
            assert get_response.headers["X-Trace-Id"] == trace_id

            document_payload = get_response.json()
            assert document_payload["id"] == str(document_id)
            assert document_payload["project_id"] == str(project_id)
            assert document_payload["original_filename"] == "document-status.pdf"
            assert document_payload["document_type"] == "construction_contract"
            assert document_payload["status"] == "uploaded"
            assert document_payload["processed_at"] is None
            assert document_payload["error_message"] is None

            missing_trace_id = f"missing-document-{uuid4()}"

            missing_document_response = await client.get(
                f"/documents/{uuid4()}",
                headers={"X-Trace-Id": missing_trace_id},
            )

            assert missing_document_response.status_code == 404
            assert missing_document_response.headers["X-Trace-Id"] == missing_trace_id
            assert missing_document_response.json()["detail"] == "Document not found."
    finally:
        if storage_key is not None:
            storage_path = STORAGE_ROOT / storage_key
            if storage_path.exists():
                storage_path.unlink()

        async with AsyncSessionLocal.begin() as session:
            if document_id is not None:
                audit_result = await session.execute(
                    select(AuditEvent).where(AuditEvent.document_id == document_id)
                )
                for audit_event in audit_result.scalars():
                    await session.delete(audit_event)

                document = await session.get(Document, document_id)
                if document is not None:
                    await session.delete(document)

            if project_id is not None:
                audit_result = await session.execute(
                    select(AuditEvent).where(AuditEvent.project_id == project_id)
                )
                for audit_event in audit_result.scalars():
                    await session.delete(audit_event)

                project = await session.get(Project, project_id)
                if project is not None:
                    await session.delete(project)
  
                    
@pytest.mark.asyncio
async def test_process_missing_stored_file_marks_document_failed() -> None:
    """Missing stored bytes produce a failed document and auditable evidence."""
    trace_id = f"missing-file-{uuid4()}"
    project_name = f"Missing Storage File Test {uuid4()}"
    pdf_bytes = build_text_pdf()

    transport = httpx.ASGITransport(app=app)
    project_id: UUID | None = None
    document_id: UUID | None = None
    storage_key: str | None = None

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            project_response = await client.post(
                "/projects",
                headers={"X-Trace-Id": trace_id},
                json={
                    "name": project_name,
                    "description": "Created by the missing storage file integration test.",
                },
            )
            assert project_response.status_code == 201
            project_id = UUID(project_response.json()["id"])

            upload_response = await client.post(
                f"/projects/{project_id}/documents",
                headers={"X-Trace-Id": trace_id},
                data={"document_type": "construction_contract"},
                files={
                    "file": (
                        "missing-storage-file.pdf",
                        pdf_bytes,
                        "application/pdf",
                    )
                },
            )
            assert upload_response.status_code == 201

            uploaded_document = upload_response.json()
            document_id = UUID(uploaded_document["id"])
            storage_key = uploaded_document["storage_key"]

            storage_path = STORAGE_ROOT / storage_key
            assert storage_path.is_file()

            storage_path.unlink()
            assert not storage_path.exists()

            process_response = await client.post(
                f"/documents/{document_id}/process",
                headers={"X-Trace-Id": trace_id},
            )

            expected_error_message = f"Stored document file not found: {storage_key}"

            assert process_response.status_code == 422
            assert process_response.headers["X-Trace-Id"] == trace_id
            assert process_response.json()["detail"] == expected_error_message

        async with AsyncSessionLocal() as session:
            document = await session.get(Document, document_id)
            assert document is not None
            assert document.status == DocumentStatus.FAILED
            assert document.processed_at is None
            assert document.error_message == expected_error_message

            page_result = await session.execute(
                select(DocumentPage).where(DocumentPage.document_id == document_id)
            )
            assert list(page_result.scalars().all()) == []

            chunk_result = await session.execute(
                select(Chunk)
                .join(DocumentPage)
                .where(DocumentPage.document_id == document_id)
            )
            assert list(chunk_result.scalars().all()) == []

            started_result = await session.execute(
                select(AuditEvent).where(
                    AuditEvent.document_id == document_id,
                    AuditEvent.action == AuditAction.DOCUMENT_PROCESSING_STARTED,
                    AuditEvent.trace_id == trace_id,
                )
            )
            started_event = started_result.scalar_one()
            assert started_event.event_data == {"storage_key": storage_key}

            failed_result = await session.execute(
                select(AuditEvent).where(
                    AuditEvent.document_id == document_id,
                    AuditEvent.action == AuditAction.DOCUMENT_FAILED,
                    AuditEvent.trace_id == trace_id,
                )
            )
            failed_event = failed_result.scalar_one()
            assert failed_event.project_id == project_id
            assert failed_event.event_data == {"reason": expected_error_message}
    finally:
        if storage_key is not None:
            storage_path = STORAGE_ROOT / storage_key
            if storage_path.exists():
                storage_path.unlink()

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

                audit_result = await session.execute(
                    select(AuditEvent).where(AuditEvent.document_id == document_id)
                )
                for audit_event in audit_result.scalars():
                    await session.delete(audit_event)

                document = await session.get(Document, document_id)
                if document is not None:
                    await session.delete(document)

            if project_id is not None:
                audit_result = await session.execute(
                    select(AuditEvent).where(AuditEvent.project_id == project_id)
                )
                for audit_event in audit_result.scalars():
                    await session.delete(audit_event)

                project = await session.get(Project, project_id)
                if project is not None:
                    await session.delete(project)


@pytest.mark.asyncio
async def test_process_pdf_above_page_limit_marks_document_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PDF above the configured page limit fails without saved pages or chunks."""
    trace_id = f"page-limit-{uuid4()}"
    project_name = f"Page Limit Test {uuid4()}"
    pdf_bytes = build_three_page_text_pdf()
    expected_error_message = "PDF page count exceeds the configured limit of 2."

    monkeypatch.setattr(
        "app.api.routes.processing.get_settings",
        lambda: SimpleNamespace(
            max_pdf_page_count=2,
            max_pdf_chunk_count=5_000,
        ),
    )

    transport = httpx.ASGITransport(app=app)
    project_id: UUID | None = None
    document_id: UUID | None = None
    storage_key: str | None = None

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            project_response = await client.post(
                "/projects",
                headers={"X-Trace-Id": trace_id},
                json={
                    "name": project_name,
                    "description": "Created by the page limit integration test.",
                },
            )
            assert project_response.status_code == 201
            project_id = UUID(project_response.json()["id"])

            upload_response = await client.post(
                f"/projects/{project_id}/documents",
                headers={"X-Trace-Id": trace_id},
                data={"document_type": "construction_contract"},
                files={
                    "file": (
                        "over-page-limit.pdf",
                        pdf_bytes,
                        "application/pdf",
                    )
                },
            )
            assert upload_response.status_code == 201

            uploaded_document = upload_response.json()
            document_id = UUID(uploaded_document["id"])
            storage_key = uploaded_document["storage_key"]

            process_response = await client.post(
                f"/documents/{document_id}/process",
                headers={"X-Trace-Id": trace_id},
            )

            assert process_response.status_code == 422
            assert process_response.headers["X-Trace-Id"] == trace_id
            assert process_response.json()["detail"] == expected_error_message

        async with AsyncSessionLocal() as session:
            document = await session.get(Document, document_id)
            assert document is not None
            assert document.status == DocumentStatus.FAILED
            assert document.processed_at is None
            assert document.error_message == expected_error_message

            page_result = await session.execute(
                select(DocumentPage).where(DocumentPage.document_id == document_id)
            )
            assert list(page_result.scalars().all()) == []

            chunk_result = await session.execute(
                select(Chunk)
                .join(DocumentPage)
                .where(DocumentPage.document_id == document_id)
            )
            assert list(chunk_result.scalars().all()) == []

            failed_result = await session.execute(
                select(AuditEvent).where(
                    AuditEvent.document_id == document_id,
                    AuditEvent.action == AuditAction.DOCUMENT_FAILED,
                    AuditEvent.trace_id == trace_id,
                )
            )
            failed_event = failed_result.scalar_one()
            assert failed_event.project_id == project_id
            assert failed_event.event_data == {"reason": expected_error_message}
    finally:
        if storage_key is not None:
            storage_path = STORAGE_ROOT / storage_key
            if storage_path.exists():
                storage_path.unlink()

        async with AsyncSessionLocal.begin() as session:
            if document_id is not None:
                audit_result = await session.execute(
                    select(AuditEvent).where(AuditEvent.document_id == document_id)
                )
                for audit_event in audit_result.scalars():
                    await session.delete(audit_event)

                document = await session.get(Document, document_id)
                if document is not None:
                    await session.delete(document)

            if project_id is not None:
                audit_result = await session.execute(
                    select(AuditEvent).where(AuditEvent.project_id == project_id)
                )
                for audit_event in audit_result.scalars():
                    await session.delete(audit_event)

                project = await session.get(Project, project_id)
                if project is not None:
                    await session.delete(project)
                    
@pytest.mark.asyncio
async def test_process_pdf_above_chunk_limit_marks_document_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A PDF above the configured chunk limit fails without saved pages or chunks."""
    trace_id = f"chunk-limit-{uuid4()}"
    project_name = f"Chunk Limit Test {uuid4()}"
    pdf_bytes = build_text_pdf()
    expected_error_message = "PDF chunk count exceeds the configured limit of 1."

    monkeypatch.setattr(
        "app.api.routes.processing.get_settings",
        lambda: SimpleNamespace(
            max_pdf_page_count=250,
            max_pdf_chunk_count=1,
        ),
    )

    transport = httpx.ASGITransport(app=app)
    project_id: UUID | None = None
    document_id: UUID | None = None
    storage_key: str | None = None

    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            project_response = await client.post(
                "/projects",
                headers={"X-Trace-Id": trace_id},
                json={
                    "name": project_name,
                    "description": "Created by the chunk limit integration test.",
                },
            )
            assert project_response.status_code == 201
            project_id = UUID(project_response.json()["id"])

            upload_response = await client.post(
                f"/projects/{project_id}/documents",
                headers={"X-Trace-Id": trace_id},
                data={"document_type": "construction_contract"},
                files={
                    "file": (
                        "over-chunk-limit.pdf",
                        pdf_bytes,
                        "application/pdf",
                    )
                },
            )
            assert upload_response.status_code == 201

            uploaded_document = upload_response.json()
            document_id = UUID(uploaded_document["id"])
            storage_key = uploaded_document["storage_key"]

            process_response = await client.post(
                f"/documents/{document_id}/process",
                headers={"X-Trace-Id": trace_id},
            )

            assert process_response.status_code == 422
            assert process_response.headers["X-Trace-Id"] == trace_id
            assert process_response.json()["detail"] == expected_error_message

        async with AsyncSessionLocal() as session:
            document = await session.get(Document, document_id)
            assert document is not None
            assert document.status == DocumentStatus.FAILED
            assert document.processed_at is None
            assert document.error_message == expected_error_message

            page_result = await session.execute(
                select(DocumentPage).where(DocumentPage.document_id == document_id)
            )
            assert list(page_result.scalars().all()) == []

            chunk_result = await session.execute(
                select(Chunk)
                .join(DocumentPage)
                .where(DocumentPage.document_id == document_id)
            )
            assert list(chunk_result.scalars().all()) == []

            failed_result = await session.execute(
                select(AuditEvent).where(
                    AuditEvent.document_id == document_id,
                    AuditEvent.action == AuditAction.DOCUMENT_FAILED,
                    AuditEvent.trace_id == trace_id,
                )
            )
            failed_event = failed_result.scalar_one()
            assert failed_event.project_id == project_id
            assert failed_event.event_data == {"reason": expected_error_message}
    finally:
        if storage_key is not None:
            storage_path = STORAGE_ROOT / storage_key
            if storage_path.exists():
                storage_path.unlink()

        async with AsyncSessionLocal.begin() as session:
            if document_id is not None:
                audit_result = await session.execute(
                    select(AuditEvent).where(AuditEvent.document_id == document_id)
                )
                for audit_event in audit_result.scalars():
                    await session.delete(audit_event)

                document = await session.get(Document, document_id)
                if document is not None:
                    await session.delete(document)

            if project_id is not None:
                audit_result = await session.execute(
                    select(AuditEvent).where(AuditEvent.project_id == project_id)
                )
                for audit_event in audit_result.scalars():
                    await session.delete(audit_event)

                project = await session.get(Project, project_id)
                if project is not None:
                    await session.delete(project)
                    

@pytest.mark.asyncio
async def test_chunk_embedding_metadata_persists() -> None:
    """A chunk stores and returns embedding metadata through pgvector."""
    project_id: UUID | None = None
    document_id: UUID | None = None
    page_id: UUID | None = None
    chunk_id: UUID | None = None

    embedding = [0.0] * 1536
    embedding[0] = 1.0
    text_content = "The contractor must provide written notice before termination."
    embedded_at = datetime.now(UTC)
    embedding_model = "text-embedding-3-small"
    embedding_text_hash = sha256(text_content.encode("utf-8")).hexdigest()

    try:
        async with AsyncSessionLocal.begin() as session:
            project = Project(
                name=f"Embedding Persistence Test {uuid4()}",
                description="Created by the embedding persistence integration test.",
            )
            session.add(project)
            await session.flush()
            project_id = project.id

            document = Document(
                project_id=project.id,
                original_filename="embedding-test.pdf",
                storage_key=f"tests/embedding-{uuid4()}.pdf",
                sha256=sha256(b"embedding-test-pdf").hexdigest(),
                content_type="application/pdf",
                size_bytes=1,
                document_type=DocumentType.CONSTRUCTION_CONTRACT,
                status=DocumentStatus.PROCESSED,
                processed_at=embedded_at,
            )
            session.add(document)
            await session.flush()
            document_id = document.id

            page = DocumentPage(
                document_id=document.id,
                page_number=1,
                text_content=text_content,
            )
            session.add(page)
            await session.flush()
            page_id = page.id

            chunk = Chunk(
                page_id=page.id,
                chunk_index=0,
                text_content=text_content,
                token_count=10,
                embedding=embedding,
                embedding_model=embedding_model,
                embedding_text_hash=embedding_text_hash,
                embedded_at=embedded_at,
            )
            session.add(chunk)
            await session.flush()
            chunk_id = chunk.id

        async with AsyncSessionLocal() as session:
            stored_chunk = await session.get(Chunk, chunk_id)

            assert stored_chunk is not None
            assert stored_chunk.embedding is not None
            assert len(stored_chunk.embedding) == 1536
            assert stored_chunk.embedding[0] == 1.0
            assert stored_chunk.embedding[1] == 0.0
            assert stored_chunk.embedding_model == embedding_model
            assert stored_chunk.embedding_text_hash == embedding_text_hash
            assert stored_chunk.embedded_at == embedded_at
    finally:
        async with AsyncSessionLocal.begin() as session:
            if chunk_id is not None:
                chunk = await session.get(Chunk, chunk_id)
                if chunk is not None:
                    await session.delete(chunk)

            if page_id is not None:
                page = await session.get(DocumentPage, page_id)
                if page is not None:
                    await session.delete(page)

            if document_id is not None:
                document = await session.get(Document, document_id)
                if document is not None:
                    await session.delete(document)

            if project_id is not None:
                project = await session.get(Project, project_id)
                if project is not None:
                    await session.delete(project)
                    