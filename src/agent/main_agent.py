import asyncio
import logging
import sys
from typing import Optional
from src.agent.graph_builder.agent_state import AgentState
from src.agent.graph_builder.compiled_agent import get_compiled_agent, close_checkpointer

logger = logging.getLogger("main_agent")


async def main_agent(
    user_id: str, # Required - pass test UUID during testing
    query: str,
    return_sources: bool = False,
    thread_id: str = "default"
) -> dict:
    """
    Main function to ask a tax question using the multi-agent system with memory.

    Args:
        user_id: User ID for tracking (required - use test UUID for testing)
        query: User's question
        return_sources: Whether to include source documents
        thread_id: Conversation thread ID for memory (default: "default")
    
    Returns:
        Dictionary with answer and optional sources
    
    Example:
        >>> # Testing with a test UUID
        >>> result = await main_agent(
        ...     user_id="test-user-123",
        ...     query="How is PAYE calculated?"
        ... )
        >>> print(result['answer'])
        
        >>> # Production with real user ID
        >>> result = await main_agent(
        ...     user_id="user_12345",
        ...     query="What is VAT?",
        ...     thread_id="session_1"
        ... )
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
    
    # Configuration for memory
    config = {"configurable": {"thread_id": thread_id}}
    
    # Run the graph
    final_state = await app.ainvoke(initial_state, config=config)
    
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


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# async def main():
#     """Example usage of the multi-agent system."""
    
#     print("="*70)
#     print("MULTI-AGENT TAX SYSTEM (LangGraph)")
#     print("="*70)
    
#     # Example 1: Tax policy question (routes to tax agent)
#     print("\n" + "="*70)
#     print("EXAMPLE 1: Tax Policy Question")
#     print("="*70)
    
#     result1 = await main_agent(
#         "What is VAT in Nigeria?",
#         return_sources=True
#     )
    
#     print(f"\nQuestion: What is VAT in Nigeria?")
#     print(f"Route: {result1['route_used']}")
#     print(f"Model: {result1['model_used']}")
#     print(f"\nAnswer:\n{result1['answer'][:300]}...")
    
#     # Example 2: PAYE question (routes to PAYE agent)
#     print("\n" + "="*70)
#     print("EXAMPLE 2: PAYE Calculation Question")
#     print("="*70)
    
#     result2 = await main_agent(
#         "How is PAYE calculated for a salary of 500,000 Naira?",
#         return_sources=True
#     )
    
#     print(f"\nQuestion: How is PAYE calculated for 500k salary?")
#     print(f"Route: {result2['route_used']}")
#     print(f"Model: {result2['model_used']}")
#     print(f"\nAnswer:\n{result2['answer'][:300]}...")
    
#     # Example 3: Combined question (routes to both agents)
#     print("\n" + "="*70)
#     print("EXAMPLE 3: Combined Question")
#     print("="*70)
    
#     result3 = await main_agent(
#         "What tax reliefs are available and how do they affect PAYE calculation?",
#         return_sources=True
#     )
    
#     print(f"\nQuestion: Tax reliefs and PAYE?")
#     print(f"Route: {result3['route_used']}")
#     print(f"Model: {result3['model_used']}")
#     print(f"\nAnswer:\n{result3['answer'][:300]}...")
    
#     print("\n" + "="*70)
#     print("✅ All examples completed!")
#     print("="*70)
    
#     # Cleanup: Close database connection pool
#     await close_checkpointer()


# if __name__ == '__main__':

#     if sys.platform == "win32":
#         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#     asyncio.run(main())
