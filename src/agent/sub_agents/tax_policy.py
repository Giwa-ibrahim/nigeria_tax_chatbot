import logging
from src.agent.graph_builder.agent_state import AgentState
from src.tools.rag import query_rag
from src.agent.utils import format_chat_history
from src.tools.web_search import search_web

logger = logging.getLogger("tax_policy_agent")


async def tax_policy_agent(state: AgentState) -> AgentState:
    """
    Tax Policy Agent - Handles general tax questions.
    Enriched with web search for latest updates.
    """
    logger.info("📚 Tax Policy Agent processing...")
    
    query = state["query"]
    
    # Search the web for latest information
    logger.info("🔍 Searching web for latest tax policy updates...")
    web_results = search_web(query, max_results=3)
    
    # Prepare enriched query context if web results are available
    enriched_query = query
    if web_results:
        enriched_query = f"{query}\n\nLatest official information:\n{web_results}"
        logger.info("✅ Query enriched with web search results")
    
    # Query RAG for document-based context
    result = query_rag(
        user_query=enriched_query,
        collection_type="tax",
        top_k=3,
        return_sources=True,
        chat_history=format_chat_history(state.get("messages", []))  
    )
    
    # Use the RAG answer directly (web context already integrated)
    state["tax_answer"] = result["answer"]
    state["model_used"] = result["model_used"]
    
    # Add sources if not already present
    if "sources" not in state or not state["sources"]:
        state["sources"] = result.get("sources", [])
    else:
        state["sources"].extend(result.get("sources", []))
    
    logger.info("✅ Tax Policy Agent completed")
    return state

