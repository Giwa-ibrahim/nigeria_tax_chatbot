import logging
from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row

# Import agent state and functions
from src.agent.graph_builder.agent_state import AgentState
from src.agent.sub_agents.router import route_query
from src.agent.sub_agents.tax_policy import tax_policy_agent
from src.agent.sub_agents.paye import paye_calculation_agent
from src.agent.sub_agents.combined_agent import combined_agent
from src.agent.sub_agents.financial_advice import financial_advice_agent
from src.agent.response_generator import response_generator, decide_next_step
from src.configurations.config import settings

logger = logging.getLogger("compiled_agent")

_checkpointer = None
_connection_pool = None
compiled_agent = None
db_initialized = False


async def get_checkpointer():
    """Get or create a PostgreSQL checkpointer with connection pooling."""
    global _checkpointer, _connection_pool, db_initialized
    
    # Return existing instance if already initialized
    if db_initialized and _checkpointer is not None:
        return _checkpointer
    
    db_url = settings.DATABASE_URL
    
    if not db_url:
        logger.warning("⚠️ DATABASE_URL not found. Running without memory...")
        return None
    
    try:
        logger.info("Initializing database connection pool...")
        
        # Create connection pool
        connection_kwargs = {
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row
        }
        
        _connection_pool = AsyncConnectionPool(
            conninfo=db_url,
            max_size=20,
            kwargs=connection_kwargs,
            open=False
        )
        
        await _connection_pool.open()
        logger.info("✅ Database connection pool established")
        
        # Initialize checkpointer
        _checkpointer = AsyncPostgresSaver(conn=_connection_pool)
        await _checkpointer.setup()
        
        # Mark DB as initialized
        db_initialized = True
        logger.info("✅ Checkpointer setup completed")
        
        return _checkpointer
        
    except Exception as e:
        logger.error(f"❌ Error setting up checkpointer: {e}")
        await close_checkpointer()
        raise e


async def get_compiled_agent():
    """
    Get the compiled agent (initializes once and reuses).
    """
    global compiled_agent
    
    # Return cached agent if already compiled
    if compiled_agent is not None:
        return compiled_agent
    
    # Initialize checkpointer first
    checkpointer = await get_checkpointer()
    
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
    
    # Compile graph once and cache it
    compiled_agent = workflow.compile(checkpointer=checkpointer)
    logger.info("✅ Agent compiled and ready")
    
    return compiled_agent


async def close_checkpointer():
    """Close the database connection pool and cleanup resources."""
    global _checkpointer, _connection_pool, compiled_agent, db_initialized
    
    try:
        if _connection_pool:
            await _connection_pool.close()
            logger.info("✅ Database connection pool closed")
    except Exception as e:
        logger.error(f"❌ Error closing connection pool: {e}")
    finally:
        _checkpointer = None
        _connection_pool = None
        compiled_agent = None
        db_initialized = False
