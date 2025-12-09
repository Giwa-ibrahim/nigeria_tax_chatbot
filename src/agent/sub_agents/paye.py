import logging
from src.agent.graph_builder.agent_state import AgentState
from src.tools.rag import query_rag
from src.agent.utils import format_chat_history

logger = logging.getLogger("paye_agent")


async def paye_calculation_agent(state: AgentState) -> AgentState:
    """
    PAYE Calculation Agent - Handles PAYE-specific questions.
    """
    logger.info("💰 PAYE Calculation Agent processing...")
    
    result = query_rag(
        user_query=state["query"],
        collection_type="paye",
        top_k=3,
        return_sources=True,
        chat_history=format_chat_history(state.get("messages", []))
    )
    
    state["paye_answer"] = result["answer"]
    state["model_used"] = result["model_used"]
    
    # Add sources if not already present
    if "sources" not in state or not state["sources"]:
        state["sources"] = result.get("sources", [])
    else:
        state["sources"].extend(result.get("sources", []))
    
    logger.info("✅ PAYE Calculation Agent completed")
    return state
