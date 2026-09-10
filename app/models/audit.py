import enum
from uuid import UUID

from sqlalchemy import JSON, Enum, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampedUUIDModel


def enum_values(enum_class: type[enum.StrEnum]) -> list[str]:
    """Return enum values rather than Python member names for PostgreSQL."""
    return [member.value for member in enum_class]


class AuditAction(enum.StrEnum):
    """Actions recorded for document-processing traceability."""

    PROJECT_CREATED = "project_created"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_PROCESSING_STARTED = "document_processing_started"
    DOCUMENT_PROCESSED = "document_processed"
    DOCUMENT_FAILED = "document_failed"
    DOCUMENT_DELETED = "document_deleted"


class AuditEvent(TimestampedUUIDModel, Base):
    """Immutable event record for the initial audit trail."""

    __tablename__ = "audit_events"

    project_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    document_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    action: Mapped[AuditAction] = mapped_column(
        Enum(
            AuditAction,
            name="audit_action",
            values_callable=enum_values,
        ),
        nullable=False,
    )
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    event_data: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)