from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from openai import OpenAIError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.db.session import get_db_session
from app.models.project import Project
from app.schemas.evidence import EvidenceSearchRequest, EvidenceSearchResponse
from app.services.embeddings import EmbeddingValidationError
from app.services.openai_embeddings import OpenAIEmbeddingClient
from app.services.query_retrieval import search_project_evidence

router = APIRouter(prefix="/projects/{project_id}", tags=["evidence"])


@router.post(
    "/evidence-search",
    response_model=EvidenceSearchResponse,
    summary="Retrieve citation-ready evidence for a project question",
    description=(
        "Embeds a question and returns only current, processed, project-scoped "
        "evidence with page-citable source locations."
    ),
)
async def evidence_search(
    project_id: UUID,
    payload: EvidenceSearchRequest,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> EvidenceSearchResponse:
    """Return citation-ready evidence without generating an answer."""
    del request

    project = await session.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found.",
        )

    settings = get_settings()
    api_key = settings.openai_api_key

    if api_key is None or not api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service is not configured.",
        )

    try:
        client = OpenAIEmbeddingClient(api_key)
        return await search_project_evidence(
            project_id,
            question=payload.question,
            session=session,
            client=client,
            embedding_model=settings.openai_embedding_model,
            embedding_dimensions=settings.embedding_dimensions,
            limit=payload.limit,
        )
    except EmbeddingValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service returned an invalid response.",
        ) from exc
    except OpenAIError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to retrieve evidence at this time.",
        ) from exc