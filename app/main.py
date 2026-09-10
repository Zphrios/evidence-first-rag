from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.core.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Evidence-first RAG service for construction-contract and payment-document review. "
        "It returns page-cited evidence, routes uncertainty to human review, and avoids "
        "unsupported conclusions."
    ),
)

app.include_router(health_router)


@app.get(
    "/",
    summary="Get service information",
    description="Returns a short service description and the OpenAPI documentation path.",
)
async def root() -> dict[str, str]:
    """Return basic service metadata."""
    return {
        "message": "Evidence-First RAG Service is running.",
        "docs": "/docs",
    }