"""API and web routes for The Empathy Engine."""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.core.config import settings
from app.models.schemas import GenerateSpeechRequest, GenerateSpeechResponse
from app.services.pipeline import EmpathyPipeline, get_pipeline

router = APIRouter()
templates = Jinja2Templates(directory=settings.templates_dir)


def pipeline_dependency() -> EmpathyPipeline:
    """Build the application pipeline and convert setup issues to API errors."""
    try:
        return get_pipeline()
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except LookupError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="NLTK VADER data is missing. Run: python -m nltk.downloader vader_lexicon",
        ) from exc


@router.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """Render the browser UI."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@router.get("/health")
async def health_check() -> dict[str, str]:
    """Return service health for deployment checks."""
    return {"status": "ok", "service": settings.app_name}


@router.post(
    "/generate",
    response_model=GenerateSpeechResponse,
    status_code=status.HTTP_201_CREATED,
)
async def generate_speech(
    payload: GenerateSpeechRequest,
    pipeline: EmpathyPipeline = Depends(pipeline_dependency),
) -> GenerateSpeechResponse:
    """Generate normal and emotional speech from the supplied text."""
    try:
        return await pipeline.generate(payload)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
