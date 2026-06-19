import logging
from src.agent.graph_builder.agent_state import AgentState
from src.agent.graph_builder.compiled_agent import get_compiled_agent
from src.agent.context_enrichment import enrich_agent_state

logger = logging.getLogger("main_agent")


async def main_agent(
    user_id: str, # Required - pass test UUID during testing
    query: str,
    return_sources: bool = False,
    thread_id: str = "default"
) -> dict:
    """
    Main function to ask a tax question using the multi-agent system with memory.
    
    🆕 PERSONALIZATION: Now enriches agent state with user data from main app!
    - Pre-loads tax calculation history for instant PAYE calculations
    - Includes income and expense data for financial advice
    - Uses display name for personalized greetings

    Args:
        user_id: User ID for tracking (required - use test UUID for testing)
        query: User's question
        return_sources: Whether to include source documents
        thread_id: Conversation thread ID for memory (default: "default")
    
    Returns:
        Dictionary with answer and optional sources

    """
    logger.info(f"❓ Question: {query} | User: {user_id}")
    
    # Get the compiled agent (initializes once, reuses after)
    app = await get_compiled_agent()
    
    # Initial state
    initial_state = {
        "user_id": user_id,
        "query": query,
        "route": "",
        "tax_answer": "",
        "paye_answer": "",
        "final_answer": "",
        "sources": [],
        "model_used": ""
    }
    
    # 🆕 ENRICH STATE WITH USER DATA FROM MAIN APP
    try:
        logger.info("📋 Enriching agent state with user data...")
        enriched_state = await enrich_agent_state(initial_state, user_id)
        
        if enriched_state.get("has_user_data"):
            logger.info("✅ User data loaded successfully!")
        else:
            logger.info("⚠️  No user data found - using standard flow")
            enriched_state = initial_state
            
    except Exception as e:
        logger.warning(f"⚠️  Failed to enrich state: {e} - using standard flow")
        enriched_state = initial_state
    
    # Configuration for memory
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the graph with enriched state
    final_state = await app.ainvoke(enriched_state, config=config)
    
    # Prepare response
    response = {
        "answer": final_state["final_answer"],
        "model_used": final_state["model_used"],
        "route_used": final_state["route"]
    }
    
    # Include sources if requested
    if return_sources and final_state.get("sources"):
        response["sources"] = final_state["sources"]
    
    logger.info(f"✅ Answer generated using route: {final_state['route']}")
    
    return response
