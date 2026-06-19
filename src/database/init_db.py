"""
Database initialization script.
Run this to create all tables in PostgreSQL.

Usage:
    python -m src.database.init_db          # Create tables
    python -m src.database.init_db --fresh  # Drop all & recreate
"""
import asyncio
import logging
import sys
from src.database.connection import init_database, health_check

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("init_db")


async def drop_all_tables():
    """Drop all existing tables."""
    from src.database.connection import get_async_engine
    from src.database.models import Base
    
    engine = get_async_engine()
    logger.warning("⚠️  Dropping all existing tables...")
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    logger.info("✅ All tables dropped")


async def main():
    """Initialize database tables."""
    try:
        # Check for --fresh flag
        fresh_install = "--fresh" in sys.argv or "-f" in sys.argv
        
        if fresh_install:
            logger.info("Starting FRESH database initialization...")
            await drop_all_tables()
        else:
            logger.info("Starting database initialization...")
       
        # Create all tables
        await init_database()
        
        # Verify database is accessible
        is_healthy = await health_check()
        
        if is_healthy:
            logger.info("✅ Database initialization completed successfully!")
            logger.info("✅ Health check passed")
            logger.info("\n📊 Tables created:")
            logger.info("  - chat_sessions")
            logger.info("  - chat_messages")
            logger.info("  - chat_summaries")
            logger.info("  - chat_users")
            return 0
        else:
            logger.error("❌ Database health check failed")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
