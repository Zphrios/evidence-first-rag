import hashlib
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db_session
from app.models.audit import AuditAction, AuditEvent
from app.models.document import Document, DocumentStatus, DocumentType
from app.models.project import Project
from app.schemas.document import DocumentRead
from app.services.storage import (
    build_document_storage_key,
    delete_document_bytes,
    save_document_bytes,
)

router = APIRouter(prefix="/projects/{project_id}/documents", tags=["documents"])


@router.post(
    "",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a source PDF to a project",
)
async def upload_document(
    project_id: UUID,
    request: Request,
    document_type: DocumentType = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_db_session),
) -> Document:
    """Validate, persist, and record a PDF upload with an audit event."""
    settings = get_settings()
    trace_id = request.state.trace_id

    project = await session.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found.",
        )

    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Uploaded file is empty.",
        )

    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Uploaded file exceeds the configured size limit.",
        )

    if not content.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Uploaded content is not a valid PDF file.",
        )

    sha256 = hashlib.sha256(content).hexdigest()

    duplicate_result = await session.execute(
        select(Document.id).where(
            Document.project_id == project_id,
            Document.sha256 == sha256,
        )
    )
    if duplicate_result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This document has already been uploaded to the project.",
        )

    document_id = uuid4()
    storage_key = build_document_storage_key(project_id, document_id)

    document = Document(
        id=document_id,
        project_id=project_id,
        original_filename=filename,
        storage_key=storage_key,
        sha256=sha256,
        content_type="application/pdf",
        size_bytes=len(content),
        document_type=document_type,
        status=DocumentStatus.UPLOADED,
    )

    audit_event = AuditEvent(
        project_id=project_id,
        document_id=document_id,
        action=AuditAction.DOCUMENT_UPLOADED,
        trace_id=trace_id,
        event_data={
            "original_filename": filename,
            "sha256": sha256,
            "size_bytes": len(content),
            "document_type": document_type.value,
        },
    )

    file_saved = False

    try:
        save_document_bytes(storage_key, content)
        file_saved = True

        session.add(document)
        await session.flush()

        session.add(audit_event)
        await session.commit()
        await session.refresh(document)
    except IntegrityError as exc:
        await session.rollback()
        if file_saved:
            delete_document_bytes(storage_key)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This document has already been uploaded to the project.",
        ) from exc
    except SQLAlchemyError as exc:
        await session.rollback()
        if file_saved:
            delete_document_bytes(storage_key)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to store document metadata at this time.",
        ) from exc
    finally:
        await file.close()

    return document