import logging
from typing import List, Dict, Optional
from urllib.parse import urlparse
from langchain_tavily import TavilySearch
from src.configurations.config import settings

logger = logging.getLogger("tavily_tool")


def get_tavily(max_results: int = 3) -> Optional[TavilySearch]:
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
        tool = TavilySearch(
            max_results=max_results,
            tavily_api_key=settings.TAVILY_API_KEY,
            search_depth="advanced",
            include_domains=[
                "firs.gov.ng",
                "nigeriataxai.com",
                "jtb.gov.ng",
                "noa.gov.ng",
                "cbn.gov.ng",
                "cbn.gov.ng/FinInc/FinLit/"
            ]
        )
        logger.info("✅ Tavily search tool initialized")
        return tool
    except Exception as e:
        logger.error(f"Error initializing Tavily tool: {e}")
        return None

def format_results(results_dict: List[Dict]) -> str:
    """Format Tavily search results for LLM consumption."""
    if not results_dict:
        return ""
    results= results_dict['results']
    
    formatted = []
    for i, result in enumerate(results, 1):
        url = result.get("url", "")
        content = result.get("content", "")
        domain = urlparse(url).netloc.replace('www.', '')
        
        formatted.append(
            f"[Source {i}: {domain}]\n"
            f"URL: {url}\n"
            f"Content: {content}\n"
        )
    
    return "\n".join(formatted)


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
    
    # Define allowed domains for strict filtering
    allowed_domains = [
        "firs.gov.ng",
        "nigeriataxai.com",
        "jtb.gov.ng",
        "noa.gov.ng"
        "cbn.gov.ng",
        "cbn.gov.ng/FinInc/FinLit/"
    ]
    
    try:
        logger.info(f"🔍 Searching web for: {query[:100]}...")
        
        # Invoke the tool
        response = tool.invoke({"query": query})
        
        # Filter results by allowed domains
        if 'results' in response:
            filtered_results = [
                result for result in response['results']
                if any(domain in result.get("url") for domain in allowed_domains)
            ]
        response['results']= filtered_results
        logger.info(f"✅ Web search completed")
        return format_results(response)
        
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return ""


# Example usage
def main():
    query = "who are exempted to pay tax in nigeria as from 2026?"
    results = search_web(query, max_results=3)
    print("Web Search Results:\n", results)
    
if __name__ == "__main__":
    main()