"""FastAPI application entry point"""

import logging
import time
from contextlib import asynccontextmanager

from config import settings
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from middleware.error_handler import error_handler_middleware
from middleware.metrics import metrics_middleware
from middleware.rate_limiter import RateLimitMiddleware
from prometheus_client import make_asgi_app
from routes import accuracy, lottery, matches
from routes import models as model_routes
from routes import odds, predictions, races, recommendations, user, websocket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Auspex API...")
    logger.info(f"Environment: {settings.API_VERSION}")

    # Load ML models into the process-wide registry. Subsequent requests
    # reuse the loaded artifacts instead of re-reading joblib pickles.
    from services.prediction_service import load_models_into_process

    load_models_into_process()

    logger.info("API ready to serve requests")

    yield

    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Auspex — sports prediction & recommendation analytics API",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rate limiting middleware
app.add_middleware(RateLimitMiddleware)

# Custom middleware
app.middleware("http")(error_handler_middleware)
app.middleware("http")(metrics_middleware)

# Include routers
app.include_router(predictions.router, prefix="/api/v1/predictions", tags=["Predictions"])
app.include_router(recommendations.router, prefix="/api/v1/recommendations", tags=["Recommendations"])
app.include_router(matches.router, prefix="/api/v1/matches", tags=["Matches"])
app.include_router(races.router, prefix="/api/v1/races", tags=["Horse Racing"])
app.include_router(odds.router, prefix="/api/v1/odds", tags=["Odds"])
app.include_router(user.router, prefix="/api/v1/user", tags=["User"])
app.include_router(model_routes.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(lottery.router, prefix="/api/v1/lottery", tags=["Lottery"])
app.include_router(accuracy.router, prefix="/api/v1/accuracy", tags=["Accuracy"])

if settings.ENABLE_WEBSOCKET:
    app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors"""
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "timestamp": time.time(),
        },
    )


# Health check
@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.API_VERSION,
        "timestamp": time.time(),
    }


# Root endpoint
@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """Root endpoint"""
    return {
        "message": "Auspex API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
