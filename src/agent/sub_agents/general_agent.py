import logging
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph_builder.agent_state import AgentState

logger = logging.getLogger("general_agent")

async def general_agent(state: AgentState) -> AgentState:
    """Passthrough node for greetings/chitchat.
    The router already set final_answer inline, so nothing to do here.
    """
    logger.info("👋 General chat - answer already set by router")
    if state.get("final_answer"):
        state["messages"] = [
            HumanMessage(content=state["query"]),
            AIMessage(content=state["final_answer"])
        ]
    return state
