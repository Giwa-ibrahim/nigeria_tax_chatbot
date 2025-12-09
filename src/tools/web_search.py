import logging
from typing import List, Dict, Optional
from langchain_community.tools.tavily_search import TavilySearchResults
from src.configurations.config import settings

logger = logging.getLogger("tavily_tool")


def get_tavily(max_results: int = 3) -> Optional[TavilySearchResults]:
    """
    Get Tavily search tool for web search.
    
    Args:
        max_results: Maximum number of results to return
        
    Returns:
        TavilySearchResults tool or None if API key not set
    """
    if not settings.TAVILY_API_KEY:
        logger.warning("TAVILY_API_KEY not set - web search disabled")
        return None
    
    try:
        tool = TavilySearchResults(
            max_results=max_results,
            api_key=settings.TAVILY_API_KEY,
            include_domains=[
                "firs.gov.ng",
                "budget.gov.ng",
                "pwc.com/ng",
                "kpmg.com/ng",
                "nairametrics.com",
                "businessday.ng"
            ]
        )
        logger.info("✅ Tavily search tool initialized")
        return tool
    except Exception as e:
        logger.error(f"Error initializing Tavily tool: {e}")
        return None


def search_web(query: str, max_results: int = 3) -> str:
    """
    Search the web using Tavily tool.
    
    Args:
        query: Search query
        max_results: Maximum results
        
    Returns:
        Formatted search results as string
    """
    tool = get_tavily(max_results)
    
    if not tool:
        return ""
    
    try:
        logger.info(f"🔍 Searching web for: {query[:100]}...")
        
        # Invoke the tool
        results = tool.invoke({"query": query})
        
        logger.info(f"✅ Web search completed")
        return format_web_results(results)
        
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return ""


def format_web_results(results: List[Dict]) -> str:
    """Format Tavily search results for LLM consumption."""
    if not results:
        return ""
    
    formatted = []
    for i, result in enumerate(results, 1):
        url = result.get("url", "")
        content = result.get("content", "")
        
        formatted.append(
            f"[Web Source {i}]\n"
            f"URL: {url}\n"
            f"Content: {content}\n"
        )
    
    return "\n".join(formatted)