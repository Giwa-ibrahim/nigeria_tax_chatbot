"""
To startup the Swagger UI, run this command:
uvicorn src.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from src.api.routes.chat_agent import router as chat_router
from src.agent.graph_builder.compiled_agent import close_checkpointer
from src.api.utilis.auth import endpoint_auth

logger = logging.getLogger("api_app")


# ============================================================================
# LIFESPAN CONTEXT MANAGER (Startup/Shutdown)
# ============================================================================

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


# ============================================================================
# CREATE FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Nigerian Tax Chatbot API",
    description= "AI-Powered Nigerian Tax Assistant",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "Nigerian Tax Chatbot Team",
        "email": "squidtech11@gmail.com"
    }
)


# ============================================================================
# MIDDLEWARE
# ============================================================================

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# INCLUDE ROUTERS
# ============================================================================

app.include_router(chat_router, dependencies=[Depends(endpoint_auth)])


# ============================================================================
# ROOT ENDPOINT
# ============================================================================

@app.get("/", include_in_schema=False)
async def root():
    """Redirect root to API docs."""
    return RedirectResponse(url="/docs")


@app.get("/ping", tags=["health"])
async def ping():
    """Simple ping endpoint."""
    return {"status": "ok", "message": "pong"}


