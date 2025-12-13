import logging
from src.agent.graph_builder.agent_state import AgentState
from src.tools.rag import query_rag
from src.agent.utils import format_chat_history
from src.tools.web_search import search_web

logger = logging.getLogger("paye_agent")


async def paye_calculation_agent(state: AgentState) -> AgentState:
    """
    PAYE Calculation Agent - Handles PAYE-specific questions.
    Enriched with web search for latest updates.
    """
    logger.info("💰 PAYE Calculation Agent processing...")
    
    query = state["query"]
    
    # Search the web for latest PAYE information
    logger.info("🔍 Searching web for latest PAYE updates...")
    web_results = search_web(query, max_results=3)
    
    # Query RAG for document-based context
    result = query_rag(
        user_query=query,
        collection_type="paye",
        top_k=3,
        return_sources=True,
        chat_history=format_chat_history(state.get("messages", []))
    )
    
    # Enrich the answer with web search context if available
    if web_results:
        enriched_answer = (
            f"{result['answer']}\n\n"
            f"**Latest Updates from Official Sources:**\n{web_results}"
        )
        state["paye_answer"] = enriched_answer
        logger.info("✅ RAG result enriched with web search")
    else:
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

