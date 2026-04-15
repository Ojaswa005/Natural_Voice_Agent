"""FastAPI entry point for The Empathy Engine."""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.services.pipeline import warmup_pipeline_dependencies


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    application = FastAPI(
        title=settings.app_name,
        description="AI-powered emotionally expressive speech generation service.",
        version=settings.app_version,
    )

    @application.on_event("startup")
    async def _warmup_models() -> None:
        await warmup_pipeline_dependencies()

    application.mount("/static", StaticFiles(directory=settings.static_dir), name="static")
    application.include_router(router)
    return application


app = create_app()
