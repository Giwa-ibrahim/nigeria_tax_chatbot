"""
To start the API server:

uvicorn src.main:app --reload --port 8080

Reason: This runs on port 8080 to avoid conflict with Chainlit (port 8000)
"""

import structlog
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from src.configurations.config import settings
from src.api.utilis.limiter import limiter

from src.api.routes.chat_agent import router as chat_router
from src.api.routes.prompts import router as prompts_router
from src.api.routes.webhook import router as webhook_router
from src.agent.graph_builder.compiled_agent import close_checkpointer
from src.api.utilis.auth import endpoint_auth

from src.configurations.logging_config import setup_structured_logging

setup_structured_logging()
logger = structlog.get_logger("fastapi_app")


# Lifespan Context Manager (Startup/Shutdown)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Starting Nigerian Tax Chatbot API...")
    logger.info("✅ API is ready to accept requests")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Nigerian Tax Chatbot API...")
    try:
        await close_checkpointer()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


# Create FastAPI App

app = FastAPI(
    title="Nigerian Tax Chatbot API",
    description= "AI-Powered Nigerian Tax Assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")] if settings.ALLOWED_ORIGINS else []

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Include Routers
app.include_router(chat_router, dependencies=[Depends(endpoint_auth)])
app.include_router(prompts_router, dependencies=[Depends(endpoint_auth)])
app.include_router(webhook_router)  # No auth - WhatsApp handles verification


# Root Endpoint

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API docs."""
    return RedirectResponse(url="/docs")


@app.get("/ping", tags=["health"])
async def ping():
    """Simple ping endpoint."""
    return {"status": "ok", "message": "pong"}


