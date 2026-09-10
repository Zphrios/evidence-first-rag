from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db_session

router = APIRouter(tags=["health"])


@router.get(
    "/health",
    summary="Check application and database health",
    description=(
        "Confirms that the service can reach PostgreSQL and that the pgvector extension "
        "is available in the configured database."
    ),
)
async def health_check(
    session: AsyncSession = Depends(get_db_session),
) -> dict[str, str]:
    """Return service metadata only after successful database and pgvector checks."""
    try:
        await session.execute(text("SELECT 1"))
        result = await session.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        vector_extension = result.scalar_one_or_none()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connectivity check failed.",
        ) from exc

    if vector_extension != "vector":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The pgvector extension is not installed.",
        )

    settings = get_settings()

    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.app_env,
        "version": settings.app_version,
        "database": "connected",
        "pgvector": "installed",
    }