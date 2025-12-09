import logging
from src.agent.graph_builder.agent_state import AgentState
from src.tools.rag import query_rag
from src.agent.utils import format_chat_history

logger = logging.getLogger("tax_policy_agent")


async def tax_policy_agent(state: AgentState) -> AgentState:
    """
    Tax Policy Agent - Handles general tax questions.
    """
    logger.info("📚 Tax Policy Agent processing...")
    
    result = query_rag(
        user_query=state["query"],
        collection_type="tax",
        top_k=3,
        return_sources=True,
        chat_history=format_chat_history(state.get("messages", []))  
    )
    
    state["tax_answer"] = result["answer"]
    state["model_used"] = result["model_used"]
    
    # Add sources if not already present
    if "sources" not in state or not state["sources"]:
        state["sources"] = result.get("sources", [])
    else:
        state["sources"].extend(result.get("sources", []))
    
    logger.info("✅ Tax Policy Agent completed")
    return state
