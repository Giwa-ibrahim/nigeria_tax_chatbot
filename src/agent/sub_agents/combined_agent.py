import logging
from src.agent.graph_builder.agent_state import AgentState
from src.tools.rag import query_rag
from src.tools.web_search import search_web
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger("combined_agent")


async def combined_agent(state: AgentState) -> AgentState:
    """
    Combined Agent - Queries both collections for complex questions.
    Enriched with web search for latest updates.
    """
    logger.info("🔄 Combined Agent processing...")
    
    query = state["query"]
    
    # Search the web for latest information
    logger.info("🔍 Searching web for latest tax updates...")
    web_results = search_web(query, max_results=3)
    
    # Query RAG for document-based context from both collections
    result = query_rag(
        user_query=query,
        collection_type="both",
        top_k=5,
        return_sources=True
    )
    
    # Enrich the rag result with web search context if available
    if web_results:
        enriched_answer = (
            f"{result['answer']}\n\n"
            f"**Latest Updates from Official Sources:**\n{web_results}"
        )
        state["final_answer"] = enriched_answer
        logger.info("✅ RAG results enriched with web search")
    else:
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
