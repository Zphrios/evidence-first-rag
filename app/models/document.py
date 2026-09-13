import enum
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampedUUIDModel

if TYPE_CHECKING:
    from app.models.project import Project


def enum_values(enum_class: type[enum.StrEnum]) -> list[str]:
    """Return enum values rather than Python member names for PostgreSQL."""
    return [member.value for member in enum_class]


class DocumentType(enum.StrEnum):
    """Supported document categories for the initial release."""

    CONSTRUCTION_CONTRACT = "construction_contract"
    PAYMENT_APPLICATION = "payment_application"


class DocumentStatus(enum.StrEnum):
    """Processing states for uploaded documents."""

    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class Document(TimestampedUUIDModel, Base):
    """Source PDF stored and processed for evidence-based review."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("project_id", "sha256", name="uq_documents_project_sha256"),
    )

    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    storage_key: Mapped[str] = mapped_column(String(500), nullable=False, unique=True)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    content_type: Mapped[str] = mapped_column(String(100), nullable=False)
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    document_type: Mapped[DocumentType] = mapped_column(
        Enum(
            DocumentType,
            name="document_type",
            values_callable=enum_values,
        ),
        nullable=False,
    )
    status: Mapped[DocumentStatus] = mapped_column(
        Enum(
            DocumentStatus,
            name="document_status",
            values_callable=enum_values,
        ),
        nullable=False,
        default=DocumentStatus.UPLOADED,
    )
    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    project: Mapped["Project"] = relationship(back_populates="documents")
    pages: Mapped[list["DocumentPage"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="DocumentPage.page_number",
    )


class DocumentPage(TimestampedUUIDModel, Base):
    """Extracted text and metadata for one page in an uploaded PDF."""

    __tablename__ = "document_pages"
    __table_args__ = (
        UniqueConstraint("document_id", "page_number", name="uq_document_pages_number"),
    )

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    page_number: Mapped[int] = mapped_column(Integer, nullable=False)
    text_content: Mapped[str] = mapped_column(Text, nullable=False)

    document: Mapped["Document"] = relationship(back_populates="pages")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="page",
        cascade="all, delete-orphan",
        lazy="selectin",
        order_by="Chunk.chunk_index",
    )


class Chunk(TimestampedUUIDModel, Base):
    """A page-aware text segment prepared for later embedding and retrieval."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("page_id", "chunk_index", name="uq_chunks_page_index"),
    )

    page_id: Mapped[UUID] = mapped_column(
        ForeignKey("document_pages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text_content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(1536),
        nullable=True,
    )
    embedding_model: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )
    embedding_text_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    embedded_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    page: Mapped["DocumentPage"] = relationship(back_populates="chunks")