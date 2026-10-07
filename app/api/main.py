"""FastAPI Application entrypoint for MedVision AI."""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.routes.health import router as health_router
from app.api.routes.analysis import router as analysis_router
from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger

logger = get_logger("medvision.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context for startup and shutdown procedures."""
    settings = get_settings()
    setup_logging(log_level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT)
    logger.info("Starting %s v%s in %s mode", settings.PROJECT_NAME, settings.VERSION, settings.ENVIRONMENT)
    logger.info("Phase 1 Foundation: Model weights and datasets not yet loaded.")
    yield
    logger.info("Shutting down %s", settings.PROJECT_NAME)


def create_application() -> FastAPI:
    """Instantiate and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        description=(
            "Evidence-grounded multimodal medical AI decision-support platform. "
            "Assists clinicians with chest X-ray review using vision models, "
            "clinical context integration, attention localization, and evidence grounding. "
            "Notice: This is a decision-support second-opinion tool, not an autonomous diagnostic system."
        ),
        lifespan=lifespan,
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handler (Prevent leaking stack traces to end users per Rule 18)
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error("Unhandled error processing %s: %s", request.url.path, str(exc), exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "An internal server error occurred while processing your request.",
                "code": "INTERNAL_SERVER_ERROR",
                "details": "Technical logs have been recorded for review.",
            },
        )

    # Include Routers
    # Direct /health for container/load balancer probes
    app.include_router(health_router)
    # Prefixed API routes
    app.include_router(health_router, prefix=settings.API_V1_STR)
    app.include_router(analysis_router, prefix=settings.API_V1_STR)

    return app


app = create_application()
