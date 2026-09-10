from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db

router = APIRouter(tags=["Health"])


@router.get("/health", status_code=status.HTTP_200_OK)
async def health_check(db: AsyncSession = Depends(get_db)):
    """Verifies API status and Neon PostgreSQL + pgvector connectivity."""
    try:
        result = await db.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        )
        extension = result.scalar()
        db_status = "connected"
        pgvector_status = "installed" if extension == "vector" else "not_installed"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
        pgvector_status = "unknown"

    return {
        "status": "healthy",
        "database": db_status,
        "pgvector": pgvector_status,
    }