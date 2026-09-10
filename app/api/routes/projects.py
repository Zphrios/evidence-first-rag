from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db_session
from app.models.audit import AuditAction, AuditEvent
from app.models.project import Project
from app.schemas.project import ProjectCreate, ProjectListResponse, ProjectRead

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post(
    "",
    response_model=ProjectRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a construction review project",
    description=(
        "Creates a project and records a project_created audit event in the same "
        "database transaction."
    ),
)
async def create_project(
    payload: ProjectCreate,
    request: Request,
    session: AsyncSession = Depends(get_db_session),
) -> Project:
    """Create a project and its audit event atomically."""
    trace_id = request.state.trace_id

    try:
        async with session.begin():
            project = Project(
                name=payload.name,
                description=payload.description,
            )
            session.add(project)
            await session.flush()

            audit_event = AuditEvent(
                project_id=project.id,
                action=AuditAction.PROJECT_CREATED,
                trace_id=trace_id,
                event_data={
                    "project_name": project.name,
                    "has_description": project.description is not None,
                },
            )
            session.add(audit_event)

        await session.refresh(project)
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Unable to create project at this time.",
        ) from exc

    return project


@router.get(
    "",
    response_model=ProjectListResponse,
    summary="List construction review projects",
)
async def list_projects(
    session: AsyncSession = Depends(get_db_session),
) -> ProjectListResponse:
    """Return all projects ordered from newest to oldest."""
    result = await session.execute(
        select(Project).order_by(Project.created_at.desc())
    )
    return ProjectListResponse(items=list(result.scalars().all()))


@router.get(
    "/{project_id}",
    response_model=ProjectRead,
    summary="Get a construction review project",
)
async def get_project(
    project_id: UUID,
    session: AsyncSession = Depends(get_db_session),
) -> Project:
    """Return one project by UUID."""
    project = await session.get(Project, project_id)

    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Project not found.",
        )

    return project