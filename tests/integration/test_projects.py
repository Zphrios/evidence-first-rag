from uuid import UUID, uuid4

import httpx
import pytest
from sqlalchemy import select

from app.db.session import AsyncSessionLocal
from app.main import app
from app.models.audit import AuditAction, AuditEvent
from app.models.project import Project


@pytest.mark.asyncio
async def test_create_project_records_audit_event() -> None:
    """Creating a project writes a matching project_created audit event."""
    trace_id = f"test-project-{uuid4()}"
    project_name = f"Integration Test Project {uuid4()}"
    payload = {
        "name": project_name,
        "description": "Created by the Project API integration test.",
    }

    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        create_response = await client.post(
            "/projects",
            headers={"X-Trace-Id": trace_id},
            json=payload,
        )

        assert create_response.status_code == 201
        assert create_response.headers["X-Trace-Id"] == trace_id

        created_project = create_response.json()
        project_id = UUID(created_project["id"])

        assert created_project["name"] == project_name
        assert created_project["description"] == payload["description"]

        get_response = await client.get(f"/projects/{project_id}")

        assert get_response.status_code == 200
        assert get_response.json()["id"] == str(project_id)

    try:
        async with AsyncSessionLocal() as session:
            audit_result = await session.execute(
                select(AuditEvent).where(
                    AuditEvent.project_id == project_id,
                    AuditEvent.action == AuditAction.PROJECT_CREATED,
                    AuditEvent.trace_id == trace_id,
                )
            )
            audit_event = audit_result.scalar_one()

            assert audit_event.event_data == {
                "project_name": project_name,
                "has_description": True,
            }
    finally:
        async with AsyncSessionLocal.begin() as session:
            audit_result = await session.execute(
                select(AuditEvent).where(AuditEvent.project_id == project_id)
            )
            for audit_event in audit_result.scalars():
                await session.delete(audit_event)

            project = await session.get(Project, project_id)
            if project is not None:
                await session.delete(project)