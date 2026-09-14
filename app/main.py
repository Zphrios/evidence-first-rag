from uuid import uuid4

from fastapi import FastAPI, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from app.api.routes.documents import router as documents_router
from app.api.routes.evidence import router as evidence_router
from app.api.routes.health import router as health_router
from app.api.routes.processing import router as processing_router
from app.api.routes.projects import router as projects_router
from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings)

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Evidence-first RAG service for construction-contract and payment-document review. "
        "It returns page-cited evidence, routes uncertainty to human review, and avoids "
        "unsupported conclusions."
    ),
)


class TraceIdMiddleware(BaseHTTPMiddleware):
    """Attach a bounded trace identifier to every request and response."""

    async def dispatch(self, request: Request, call_next) -> Response:
        incoming_trace_id = request.headers.get("X-Trace-Id", "").strip()
        trace_id = incoming_trace_id[:64] or str(uuid4())

        request.state.trace_id = trace_id
        response = await call_next(request)
        response.headers["X-Trace-Id"] = trace_id

        return response


app.add_middleware(TraceIdMiddleware)
app.include_router(health_router)
app.include_router(projects_router)
app.include_router(evidence_router)
app.include_router(documents_router)
app.include_router(processing_router)


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