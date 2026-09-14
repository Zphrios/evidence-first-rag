from uuid import UUID

from pydantic import BaseModel, Field, field_validator


class Citation(BaseModel):
    """A verifiable location within a source document."""

    chunk_id: UUID
    document_id: UUID
    document_filename: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    excerpt: str = Field(min_length=1)

    @field_validator("document_filename", "excerpt")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        """Reject empty or whitespace-only source text."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class EvidenceSearchResponse(BaseModel):
    """Evidence retrieved for a project-scoped natural-language question."""

    question: str = Field(min_length=1)
    evidence: list[Citation]
    no_evidence_found: bool

    @field_validator("question")
    @classmethod
    def reject_blank_question(cls, value: str) -> str:
        """Reject an empty or whitespace-only search question."""
        if not value.strip():
            raise ValueError("must not be blank")
        return value