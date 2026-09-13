"""add chunk embedding metadata

Revision ID: 2b4b1fa09ce7
Revises: f1a177367c7f
Create Date: 2026-09-13 22:16:13.916572
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "2b4b1fa09ce7"
down_revision: str | Sequence[str] | None = "f1a177367c7f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Add nullable embedding metadata to processed chunks."""
    op.add_column(
        "chunks",
        sa.Column("embedding", Vector(1536), nullable=True),
    )
    op.add_column(
        "chunks",
        sa.Column(
            "embedding_model",
            sa.String(length=100),
            nullable=True,
        ),
    )
    op.add_column(
        "chunks",
        sa.Column(
            "embedding_text_hash",
            sa.String(length=64),
            nullable=True,
        ),
    )
    op.add_column(
        "chunks",
        sa.Column(
            "embedded_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Remove embedding metadata from processed chunks."""
    op.drop_column("chunks", "embedded_at")
    op.drop_column("chunks", "embedding_text_hash")
    op.drop_column("chunks", "embedding_model")
    op.drop_column("chunks", "embedding")