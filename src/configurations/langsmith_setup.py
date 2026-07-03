import os, structlog
from dotenv import load_dotenv

logger = structlog.get_logger("langsmith_setup")

def setup_langsmith() -> None:
    """Configure LangSmith environment variables for automatic tracing."""
    # Ensure .env is pushed to os.environ first
    load_dotenv() 
    
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    project= "taxbot"
    
    if not api_key:
        logger.warning("LangSmith tracing disabled (no API key found in .env)")
        return
    
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project
    
    logger.info(f"LangSmith tracing enabled project")