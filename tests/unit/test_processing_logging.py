import logging
from uuid import UUID, uuid4

import httpx
import pymupdf
import pytest
from sqlalchemy import select

from app.api.routes.processing import logger
from app.db.session import AsyncSessionLocal
from app.main import app
from app.models.audit import AuditEvent
from app.models.document import Chunk, Document, DocumentPage
from app.models.project import Project
from app.services.storage import STORAGE_ROOT


def build_sensitive_text_pdf() -> bytes:
    """Create a PDF whose source text must never appear in application logs."""
    source_text = (
        "Confidential contract term: payment is due in 30 days "
        "and the approved amount is 875000."
    )

    pdf_document = pymupdf.open()
    page = pdf_document.new_page()
    page.insert_text((72, 72), source_text)

    pdf_bytes = pdf_document.tobytes()
    pdf_document.close()
    return pdf_bytes


@pytest.mark.asyncio
async def test_processing_logs_metadata_without_pdf_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Successful processing logs operational metadata without source text."""
    trace_id = f"log-test-{uuid4()}"
    project_name = f"Logging Test Project {uuid4()}"
    source_text = (
        "Confidential contract term: payment is due in 30 days "
        "and the approved amount is 875000."
    )
    pdf_bytes = build_sensitive_text_pdf()

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
                    "description": "Created by the processing logging test.",
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
                        "confidential-contract.pdf",
                        pdf_bytes,
                        "application/pdf",
                    )
                },
            )
            assert upload_response.status_code == 201

            uploaded_document = upload_response.json()
            document_id = UUID(uploaded_document["id"])
            storage_key = uploaded_document["storage_key"]

            with caplog.at_level(logging.INFO, logger=logger.name):
                process_response = await client.post(
                    f"/documents/{document_id}/process",
                    headers={"X-Trace-Id": trace_id},
                )

            assert process_response.status_code == 200

        processing_messages = [
            record.getMessage()
            for record in caplog.records
            if record.name == logger.name
        ]
        log_output = "\n".join(processing_messages)

        assert any(
            message.startswith("document_processing_started")
            for message in processing_messages
        )
        assert any(
            message.startswith("document_processing_completed")
            for message in processing_messages
        )

        assert f"trace_id={trace_id}" in log_output
        assert f"document_id={document_id}" in log_output
        assert f"project_id={project_id}" in log_output
        assert "size_bytes=" in log_output
        assert "page_count=1" in log_output
        assert "chunk_count=1" in log_output
        assert "duration_ms=" in log_output

        assert source_text not in log_output
        assert "payment is due in 30 days" not in log_output
        assert "875000" not in log_output
        assert storage_key not in log_output
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