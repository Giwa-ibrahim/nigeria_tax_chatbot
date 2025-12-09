import logging
from src.agent.graph_builder.agent_state import AgentState
from src.tools.rag import query_rag
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger("combined_agent")


async def combined_agent(state: AgentState) -> AgentState:
    """
    Combined Agent - Queries both collections for complex questions.
    """
    logger.info("🔄 Combined Agent processing...")
    
    result = query_rag(
        user_query=state["query"],
        collection_type="both",
        top_k=5,
        return_sources=True
    )
    
    state["final_answer"] = result["answer"]
    state["sources"] = result.get("sources", [])
    state["model_used"] = result["model_used"]
    
    logger.info("✅ Combined Agent completed")

    # Save conversation to messages
    state["messages"] = [
        HumanMessage(content=state["query"]),
        AIMessage(content=state["final_answer"])
    ]

    return state