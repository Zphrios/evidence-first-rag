import hashlib
from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.main import app
from app.models.audit import AuditAction, AuditEvent
from app.models.document import Document
from app.models.project import Project
from app.services.storage import STORAGE_ROOT


@pytest.mark.asyncio
async def test_upload_pdf_records_document_and_audit_event() -> None:
    """Uploading a PDF stores metadata, audit evidence, and local file bytes."""
    trace_id = f"test-document-{uuid4()}"
    project_name = f"Document Upload Test {uuid4()}"
    pdf_bytes = b"%PDF-1.4\n% integration-test document\n%%EOF\n"
    expected_sha256 = hashlib.sha256(pdf_bytes).hexdigest()

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
                    "description": "Created by the document upload integration test.",
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
                        "integration-contract.pdf",
                        pdf_bytes,
                        "application/pdf",
                    )
                },
            )

            assert upload_response.status_code == 201
            assert upload_response.headers["X-Trace-Id"] == trace_id

            document_payload = upload_response.json()
            document_id = UUID(document_payload["id"])
            storage_key = document_payload["storage_key"]

            assert document_payload["project_id"] == str(project_id)
            assert document_payload["original_filename"] == "integration-contract.pdf"
            assert document_payload["content_type"] == "application/pdf"
            assert document_payload["size_bytes"] == len(pdf_bytes)
            assert document_payload["sha256"] == expected_sha256
            assert document_payload["document_type"] == "construction_contract"
            assert document_payload["status"] == "uploaded"

            duplicate_response = await client.post(
                f"/projects/{project_id}/documents",
                headers={"X-Trace-Id": f"{trace_id}-duplicate"},
                data={"document_type": "construction_contract"},
                files={
                    "file": (
                        "duplicate-name.pdf",
                        pdf_bytes,
                        "application/pdf",
                    )
                },
            )

            assert duplicate_response.status_code == 409
            assert duplicate_response.headers["X-Trace-Id"] == f"{trace_id}-duplicate"
            assert duplicate_response.json()["detail"] == (
                "This document has already been uploaded to the project."
            )

        assert storage_key is not None
        storage_path = STORAGE_ROOT / storage_key
        assert storage_path.read_bytes() == pdf_bytes

        async with AsyncSessionLocal() as session:
            document_result = await session.execute(
                select(Document).where(Document.id == document_id)
            )
            document = document_result.scalar_one()

            assert document.project_id == project_id
            assert document.sha256 == expected_sha256
            assert document.original_filename == "integration-contract.pdf"

            audit_result = await session.execute(
                select(AuditEvent).where(
                    AuditEvent.document_id == document_id,
                    AuditEvent.action == AuditAction.DOCUMENT_UPLOADED,
                    AuditEvent.trace_id == trace_id,
                )
            )
            audit_event = audit_result.scalar_one()

            assert audit_event.project_id == project_id
            assert audit_event.event_data == {
                "original_filename": "integration-contract.pdf",
                "sha256": expected_sha256,
                "size_bytes": len(pdf_bytes),
                "document_type": "construction_contract",
            }
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