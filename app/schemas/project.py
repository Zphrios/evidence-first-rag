from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ProjectCreate(BaseModel):
    """Request body for creating a construction review project."""

    name: str = Field(
        min_length=1,
        max_length=200,
        examples=["Downtown Office Renovation"],
    )
    description: str | None = Field(
        default=None,
        examples=["Review payment applications against the construction contract."],
    )


class ProjectRead(BaseModel):
    """Public representation of a construction review project."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    created_at: datetime
    updated_at: datetime


class ProjectListResponse(BaseModel):
    """Response envelope for a project collection."""

    items: list[ProjectRead]