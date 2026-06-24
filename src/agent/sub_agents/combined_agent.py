import logging
from src.agent.graph_builder.agent_state import AgentState
from src.tools.rag import query_rag
from src.tools.web_search import search_web
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.utils import format_chat_history

logger = logging.getLogger("combined_agent")


async def combined_agent(state: AgentState) -> AgentState:
    """
    Combined Agent - Queries both collections for complex questions.
    Uses parallel search: RAG for detailed knowledge + Web for latest updates.
    """
    logger.info("🔄 Combined Agent processing...")
    
    query = state["query"]
    chat_history = format_chat_history(state.get("messages", []))
    user_preferences = state.get("user_preferences", {})
    
    # Parallel search: Web search for latest info (runs independently)
    logger.info("🔍 Searching web for latest tax updates...")
    web_results = search_web(query, max_results=3)
    
    # RAG search with ORIGINAL query (not enriched)
    logger.info("📖 Querying knowledge base (both collections)...")
    result = query_rag(
        user_query=query,  # Use original query for accurate vector search
        collection_type="both",
        top_k=4, # Retrieve more documents for combined queries
        return_sources=True,
        chat_history=chat_history,
        user_preferences=user_preferences
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
        
        state["final_answer"] = response.content
        logger.info("✅ Combined RAG + Web results")
    else:
        # No web results, use RAG only
        state["final_answer"] = result["answer"]
        logger.info("ℹ️ Using RAG-only answer (no web results)")
    
    state["sources"] = result.get("sources", [])
    state["model_used"] = result["model_used"]
    
    logger.info("✅ Combined Agent completed")

    # Save conversation to messages
    state["messages"] = [
        HumanMessage(content=state["query"]),
        AIMessage(content=state["final_answer"])
    ]

    return state
