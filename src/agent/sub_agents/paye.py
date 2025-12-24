import logging
from src.agent.graph_builder.agent_state import AgentState
from src.tools.rag import query_rag
from src.agent.utils import format_chat_history
from src.tools.web_search import search_web

logger = logging.getLogger("paye_agent")


async def paye_calculation_agent(state: AgentState) -> AgentState:
    """
    PAYE Calculation Agent - Handles PAYE-specific questions.
    Uses parallel search: RAG for detailed knowledge + Web for latest updates.
    """
    logger.info("💰 PAYE Calculation Agent processing...")
    
    query = state["query"]
    
    # Parallel search: Web search for latest info (runs independently)
    logger.info("🔍 Searching web for latest PAYE updates...")
    web_results = search_web(query, max_results=3)
    
    # RAG search with ORIGINAL query (not enriched)
    logger.info("📖 Querying knowledge base...")
    result = query_rag(
        user_query=query,  # Use original query for accurate vector search
        collection_type="paye",
        top_k=3,
        return_sources=True,
        chat_history=format_chat_history(state.get("messages", []))
    )
    
    # Combine RAG answer with web results if available
    if web_results:
        # Let the LLM intelligently combine both sources
        combined_context = f"""Based on the following information sources, provide a comprehensive and up-to-date answer:

KNOWLEDGE BASE (Detailed Information):
{result['answer']}

LATEST UPDATES (From Official Sources):
{web_results}

USER QUESTION:
{query}

Combine both sources to give an accurate, up-to-date answer. Prioritize the latest information from official sources for current rates, dates, and regulations."""
        
        # Use LLM to synthesize both sources
        from src.services.llm import LLMManager
        llm_manager = LLMManager()
        llm = llm_manager.get_llm()
        response = llm.invoke(combined_context)
        
        state["paye_answer"] = response.content
        logger.info("✅ Combined RAG + Web results")
    else:
        # No web results, use RAG only
        state["paye_answer"] = result["answer"]
        logger.info("ℹ️ Using RAG-only answer (no web results)")
    
    state["model_used"] = result["model_used"]
    
    # Add sources if not already present
    if "sources" not in state or not state["sources"]:
        state["sources"] = result.get("sources", [])
    else:
        state["sources"].extend(result.get("sources", []))
    
    logger.info("✅ PAYE Calculation Agent completed")
    return state

