import logging
from datetime import UTC, datetime
from time import perf_counter
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import delete
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db_session
from app.models.audit import AuditAction, AuditEvent
from app.models.document import Chunk, Document, DocumentPage, DocumentStatus
from app.schemas.document import DocumentProcessingResult, DocumentRead
from app.services.pdf_processing import extract_and_chunk_pdf
from app.services.storage import read_document_bytes

router = APIRouter(prefix="/documents", tags=["documents"])

logger = logging.getLogger(__name__)


async def record_processing_failure(
    session: AsyncSession,
    document_id: UUID,
    trace_id: str,
    reason: str,
) -> None:
    """Persist a failed status and matching audit event when possible."""
    try:
        document = await session.get(Document, document_id)
        if document is None:
            return

        document.status = DocumentStatus.FAILED
        document.error_message = reason

        session.add(
            AuditEvent(
                project_id=document.project_id,
                document_id=document.id,
                action=AuditAction.DOCUMENT_FAILED,
                trace_id=trace_id,
                event_data={"reason": reason},
            )
        )
        await session.commit()
    except SQLAlchemyError:
        await session.rollback()
        raise

@router.get(
    "/{document_id}",
    response_model=DocumentRead,
    summary="Get document processing status",
)
async def get_document(
    document_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> Document:
    """Return document metadata and current processing status."""
    document = await session.get(Document, document_id)

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    return document

@router.post(
    "/{document_id}/process",
    response_model=DocumentProcessingResult,
    summary="Extract PDF text into page-cited chunks",
)
async def process_document(
    document_id: UUID,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> DocumentProcessingResult:
    """Extract text page by page, persist chunks, and record processing evidence."""
    trace_id = request.state.trace_id
    started_at = perf_counter()
    settings = get_settings()

    document = await session.get(Document, document_id)
    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found.",
        )

    if document.status == DocumentStatus.PROCESSING:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document processing is already in progress.",
        )

    if document.status == DocumentStatus.PROCESSED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document has already been processed.",
        )

    storage_key = document.storage_key
    project_id = document.project_id
    size_bytes = document.size_bytes
    
    logger.info(
    "document_processing_started "
    "trace_id=%s document_id=%s project_id=%s size_bytes=%s",
    trace_id,
    document_id,
    project_id,
    size_bytes,
)

    try:
        document.status = DocumentStatus.PROCESSING
        document.error_message = None

        session.add(
            AuditEvent(
                project_id=project_id,
                document_id=document_id,
                action=AuditAction.DOCUMENT_PROCESSING_STARTED,
                trace_id=trace_id,
                event_data={"storage_key": storage_key},
            )
        )
        await session.commit()
    except SQLAlchemyError as exc:
        await session.rollback()
        logger.error(
            "document_processing_failed "
            "trace_id=%s document_id=%s project_id=%s failure_category=start "
            "duration_ms=%d",
            trace_id,
            document_id,
            project_id,
            int((perf_counter() - started_at) * 1000),
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to start document processing at this time.",
        ) from exc

    try:
        pdf_bytes = read_document_bytes(storage_key)
        extracted_pages = extract_and_chunk_pdf(
            pdf_bytes,
            max_page_count=settings.max_pdf_page_count,
             max_chunk_count=settings.max_pdf_chunk_count,
        )
    except (FileNotFoundError, ValueError) as exc:
        failure_message = str(exc)
        logger.warning(
            "document_processing_failed "
            "trace_id=%s document_id=%s project_id=%s failure_category=extraction "
            "duration_ms=%d",
            trace_id,
            document_id,
            project_id,
            int((perf_counter() - started_at) * 1000),
        )

        try:
            await record_processing_failure(
                session=session,
                document_id=document_id,
                trace_id=trace_id,
                reason=failure_message,
            )
        except SQLAlchemyError as database_exc:
            await session.rollback()
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Unable to record document processing failure.",
            ) from database_exc

        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=failure_message,
        ) from exc

    page_models: list[DocumentPage] = []
    chunk_models: list[Chunk] = []

    try:
        await session.execute(
            delete(DocumentPage).where(DocumentPage.document_id == document_id)
        )

        for extracted_page in extracted_pages:
            page_model = DocumentPage(
                document_id=document_id,
                page_number=extracted_page.page_number,
                text_content=extracted_page.text_content,
            )
            page_models.append(page_model)

        session.add_all(page_models)
        await session.flush()

        for page_model, extracted_page in zip(page_models, extracted_pages, strict=True):
            for extracted_chunk in extracted_page.chunks:
                chunk_models.append(
                    Chunk(
                        page_id=page_model.id,
                        chunk_index=extracted_chunk.chunk_index,
                        text_content=extracted_chunk.text_content,
                    )
                )

        session.add_all(chunk_models)

        document.status = DocumentStatus.PROCESSED
        document.processed_at = datetime.now(UTC)
        document.error_message = None

        session.add(
            AuditEvent(
                project_id=project_id,
                document_id=document_id,
                action=AuditAction.DOCUMENT_PROCESSED,
                trace_id=trace_id,
                event_data={
                    "page_count": len(page_models),
                    "chunk_count": len(chunk_models),
                },
            )
        )
        await session.commit()
        await session.refresh(document)
    except SQLAlchemyError as exc:
        await session.rollback()
        failure_message = "Unable to save extracted document text."
        logger.error(
            "document_processing_failed "
            "trace_id=%s document_id=%s project_id=%s failure_category=persistence "
            "duration_ms=%d",
            trace_id,
            document_id,
            project_id,
            int((perf_counter() - started_at) * 1000),
        )

        try:
            await record_processing_failure(
                session=session,
                document_id=document_id,
                trace_id=trace_id,
                reason=failure_message,
            )
        except SQLAlchemyError:
            await session.rollback()

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=failure_message,
        ) from exc
        
    logger.info(
        "document_processing_completed "
        "trace_id=%s document_id=%s project_id=%s size_bytes=%s "
        "page_count=%s chunk_count=%s duration_ms=%d",
        trace_id,
        document_id,
        project_id,
        size_bytes,
        len(page_models),
        len(chunk_models),
        int((perf_counter() - started_at) * 1000),
    )
    return DocumentProcessingResult(
        document=document,
        page_count=len(page_models),
        chunk_count=len(chunk_models),
    )