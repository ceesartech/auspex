"""FastAPI application entry point"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from contextlib import asynccontextmanager
import logging
from prometheus_client import make_asgi_app
import time

from config import settings
from routes import predictions, recommendations, matches, odds, user
from routes import models as model_routes
from routes import lottery, websocket
from middleware.rate_limiter import RateLimitMiddleware
from middleware.error_handler import error_handler_middleware
from middleware.metrics import metrics_middleware

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
    logger.info("Starting Betting System API...")
    logger.info(f"Environment: {settings.API_VERSION}")

    # Load ML models
    from services.prediction_service import PredictionService

    prediction_service = PredictionService()
    prediction_service.load_models()

    logger.info("API ready to serve requests")

    yield

    # Shutdown
    logger.info("Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Personal Sports Betting & Lottery Recommendation System API",
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
app.include_router(
    predictions.router, prefix="/api/v1/predictions", tags=["Predictions"]
)
app.include_router(
    recommendations.router, prefix="/api/v1/recommendations", tags=["Recommendations"]
)
app.include_router(matches.router, prefix="/api/v1/matches", tags=["Matches"])
app.include_router(odds.router, prefix="/api/v1/odds", tags=["Odds"])
app.include_router(user.router, prefix="/api/v1/user", tags=["User"])
app.include_router(model_routes.router, prefix="/api/v1/models", tags=["Models"])
app.include_router(lottery.router, prefix="/api/v1/lottery", tags=["Lottery"])

if settings.ENABLE_WEBSOCKET:
    app.include_router(websocket.router, prefix="/ws", tags=["WebSocket"])

# Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
):
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
        "message": "Betting System API",
        "version": settings.API_VERSION,
        "docs": "/docs",
        "health": "/health",
    }
