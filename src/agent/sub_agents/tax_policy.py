import logging
from src.agent.graph_builder.agent_state import AgentState
from src.tools.rag import query_rag
from src.agent.utils import format_chat_history
from src.agent.context_injector import build_user_context_block

logger = logging.getLogger("tax_policy_agent")


async def tax_policy_agent(state: AgentState) -> AgentState:
    """
    Tax Policy Agent - Handles general tax questions using RAG.
    """
    logger.info("📚 Tax Policy Agent processing...")
    
    query = state["query"]
    chat_history = format_chat_history(state.get("messages", []))
    user_preferences = state.get("user_preferences", {})

    # Dynamic user context injection (LLM-driven — only for personal queries)
    user_ctx = build_user_context_block(state)
    rag_context = f"{user_ctx}\n{chat_history}" if user_ctx else chat_history

    # Query knowledge base
    logger.info("📖 Querying knowledge base...")
    result = query_rag(
        user_query=query,
        collection_type="tax",
        top_k=3,
        return_sources=True,
        chat_history=rag_context,
        user_preferences=user_preferences
    )
    
    # Use RAG answer directly
    state["tax_answer"] = result["answer"]
    
    state["model_used"] = result["model_used"]
    
    # Add sources if not already present
    if "sources" not in state or not state["sources"]:
        state["sources"] = result.get("sources", [])
    else:
        state["sources"].extend(result.get("sources", []))
    
    logger.info("✅ Tax Policy Agent completed")
    return state

