import logging
from src.agent.graph_builder.compiled_agent import get_compiled_agent
from src.agent.context_preparation import ContextPreparator

logger = logging.getLogger("main_agent")


async def main_agent(
    user_id: str, # Required - pass test UUID during testing
    query: str,
    return_sources: bool = False,
    thread_id: str = "default",
    provider: str = "groq"
) -> dict:
    """
    Main agent entry with context preparation
    """
    logger.info(f"❓ Question: {query} | User: {user_id}")
    
    # Prepare full context BEFORE router
    preparator = ContextPreparator(provider)
    context = await preparator.prepare_full_context(
        user_id=user_id,
        thread_id=thread_id,
        current_query=query,
        provider=provider
    )
    
    # Get the compiled agent (initializes once, reuses after)
    app = await get_compiled_agent()
    
    # Configuration for memory
    config = {"configurable": {"thread_id": thread_id}}
    
    # Pass clean context to router
    initial_state = {
        "user_id": user_id,
        "query": query,
        "messages": context["messages"],
        "user_profile": context.get("user_profile", {}),
        "user_preferences": context.get("user_preferences", {}),
        "route": "",
        "tax_answer": "",
        "paye_answer": "",
        "final_answer": "",
        "sources": [],
        "model_used": ""
    }
    
    # Run the graph with enriched state
    final_state = await app.ainvoke(initial_state, config=config)
    
    # Prepare response
    response = {
        "answer": final_state["final_answer"],
        "model_used": final_state["model_used"],
        "route_used": final_state["route"],
        "messages": final_state.get("messages", []) # to pass to preference learner later
    }
    
    # Include sources if requested
    if return_sources and final_state.get("sources"):
        response["sources"] = final_state["sources"]
    
    logger.info(f"✅ Answer generated using route: {final_state['route']}")
    
    return response
