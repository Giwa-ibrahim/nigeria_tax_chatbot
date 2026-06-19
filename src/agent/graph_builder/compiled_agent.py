import logging
from typing import Optional
from langgraph.graph import StateGraph, END

# Import agent state and functions
from src.agent.graph_builder.agent_state import AgentState
from src.agent.sub_agents.router import route_query
from src.agent.sub_agents.tax_policy import tax_policy_agent
from src.agent.sub_agents.paye import paye_calculation_agent
from src.agent.sub_agents.combined_agent import combined_agent
from src.agent.sub_agents.financial_advice import financial_advice_agent
from src.agent.response_generator import response_generator, decide_next_step
from src.configurations.config import settings

# Import custom database modules
from src.database.connection import init_database

logger = logging.getLogger("compiled_agent")

compiled_agent = None
db_initialized = False

async def get_compiled_agent():
    """
    Get the compiled agent (initializes once and reuses).
    """
    global compiled_agent, db_initialized
    
    # Return cached agent if already compiled
    if compiled_agent is not None:
        return compiled_agent
    
    if not db_initialized:
        await init_database()
        db_initialized = True
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", route_query)
    workflow.add_node("tax_agent", tax_policy_agent)
    workflow.add_node("paye_agent", paye_calculation_agent)
    workflow.add_node("combined_agent", combined_agent)
    workflow.add_node("financial_agent", financial_advice_agent)
    workflow.add_node("responder", response_generator)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    # Add conditional edges from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        decide_next_step,
        {
            "tax_agent": "tax_agent",
            "paye_agent": "paye_agent",
            "combined_agent": "combined_agent",
            "financial_agent": "financial_agent"
        }
    )
    
    # Tax agent -> responder
    workflow.add_edge("tax_agent", "responder")
    
    # PAYE agent -> responder
    workflow.add_edge("paye_agent", "responder")
    
    # Financial agent -> end (already has final answer)
    workflow.add_edge("financial_agent", END)
    
    # Combined agent -> end (already has final answer)
    workflow.add_edge("combined_agent", END)
    
    # Responder -> end
    workflow.add_edge("responder", END)
    
    # Compile graph once and cache it (no automatic checkpointing)
    compiled_agent = workflow.compile()
    logger.info("✅ Agent compiled and ready (manual state management)")
    
    return compiled_agent
