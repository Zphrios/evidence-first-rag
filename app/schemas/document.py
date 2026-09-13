from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from app.models.document import DocumentStatus, DocumentType


class DocumentRead(BaseModel):
    """Public metadata for an uploaded source document."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    original_filename: str
    storage_key: str
    sha256: str
    content_type: str
    size_bytes: int
    document_type: DocumentType
    status: DocumentStatus
    processed_at: datetime | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime
    
    
class DocumentProcessingResult(BaseModel):
    """Result returned after synchronous PDF text extraction and chunking."""

    document: DocumentRead
    page_count: int
    chunk_count: int