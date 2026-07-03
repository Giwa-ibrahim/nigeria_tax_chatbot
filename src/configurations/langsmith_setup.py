import os, structlog
from src.configurations.config import settings

logger = structlog.get_logger()


def setup_langsmith() -> None:
    """Configure LangSmith environment variables for automatic tracing.
    """
    langsmith_enabled: bool = True if settings.LANGSMITH_API_KEY else False
    if not langsmith_enabled:
        logger.info("LangSmith tracing disabled (no API key)")
        return
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
    
    logger.info("LangSmith tracing enabled project=%s", settings.LANGSMITH_PROJECT)