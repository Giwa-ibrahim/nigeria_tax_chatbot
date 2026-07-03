"""
Database connection management with SQLAlchemy.
Handles connection pooling and session lifecycle.
"""
import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import event, text

from src.configurations.config import settings
from src.database.models import Base

logger = logging.getLogger("db_connection")

# Global engine and session factory
_engine: Optional[AsyncEngine] = None
_session_factory: Optional[async_sessionmaker] = None


def get_async_engine() -> AsyncEngine:
    """
    Get or create the async SQLAlchemy engine.
    Uses connection pooling for production efficiency.
    """
    global _engine
    
    if _engine is not None:
        return _engine
    
    # Format URL for asyncpg and strip unsupported sslmode parameters
    db_url = settings.DATABASE_URL.replace("postgresql://", "postgres://", 1).replace("postgres://", "postgresql+asyncpg://", 1)
    db_url = db_url.split("?sslmode=")[0].split("&sslmode=")[0]

    
    # Engine configuration
    engine_kwargs = {
        "url": db_url,
        "echo": False,  # Set to True for SQL query logging in development
        "pool_pre_ping": True,  # Verify connections before using
        "pool_size": 10,  # Maximum connections in pool
        "max_overflow": 10,  # Additional connections if pool is full
        "pool_recycle": 3600,  # Recycle connections after 1 hour
        "pool_timeout": 30,  # Timeout waiting for connection
        "connect_args": {
            "statement_cache_size": 0,  # Disable prepared statements for PgBouncer compatibility
        }
    }
    
    _engine = create_async_engine(**engine_kwargs)
    
    # Connection event listeners for debugging
    @event.listens_for(_engine.sync_engine, "connect")
    def receive_connect(dbapi_conn, connection_record):
        logger.debug("Database connection established")
    
    @event.listens_for(_engine.sync_engine, "checkout")
    def receive_checkout(dbapi_conn, connection_record, connection_proxy):
        logger.debug("Connection checked out from pool")
    
    # logger.info("✅ Async database engine created")
    return _engine


def get_session_factory() -> async_sessionmaker:
    """
    Get or create the async session factory.
    """
    global _session_factory
    
    if _session_factory is not None:
        return _session_factory
    
    engine = get_async_engine()
    
    _session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,  # Don't expire objects after commit
        autoflush=False,  # Manual control over flushing
        autocommit=False,  # Explicit transaction control
    )
    
    # logger.info("✅ Session factory created")
    return _session_factory


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for database sessions.
    Automatically handles commit/rollback and session cleanup.
    """
    session_factory = get_session_factory()
    session = session_factory()
    
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        await session.close()


async def init_database():
    """
    Initialize database tables.
    Creates all tables defined in models.py if they don't exist.
    """
    engine = get_async_engine()
    
    logger.info("Initializing database tables...")
    
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("✅ Database tables initialized")


async def close_database():
    """
    Close database connections gracefully.
    Should be called on application shutdown.
    """
    global _engine, _session_factory
    
    if _engine is not None:
        logger.info("Closing database connections...")
        await _engine.dispose()
        _engine = None
        _session_factory = None
        logger.info("✅ Database connections closed")


async def health_check() -> bool:
    """
    Check database connectivity.
    Returns True if connection is healthy, False otherwise.
    """
    try:
        async with get_db_session() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
