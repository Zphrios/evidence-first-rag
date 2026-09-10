from contextlib import asynccontextmanager
from fastapi import FastAPI
from sqlalchemy import text
from app.api.routes import health
from app.core.config import settings
from app.db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure pgvector extension exists
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
    yield
    # Shutdown
    await engine.dispose()


app = FastAPI(
    title=settings.PROJECT_NAME,
    version="0.1.0",
    description="Evidence-First Contract & Document Risk Assistant",
    lifespan=lifespan,
)

app.include_router(health.router)


@app.get("/")
async def root():
    return {"message": "Evidence-First RAG Service is Running", "docs": "/docs"}