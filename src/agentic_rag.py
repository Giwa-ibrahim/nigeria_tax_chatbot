import asyncio
import logging
import os, sys
from typing import TypedDict, Literal, Annotated
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from rag import query_rag
from llm import LLMManager

# Configure logger
logger= logging.getLogger("agentic_rag")


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State that flows through the agent graph."""
    query: str                          # User's question
    messages: Annotated[list, add_messages] # Messages
    route: str                          # Which agent(s) to use: "tax", "paye", or "both"
    tax_answer: str                     # Answer from tax policy agent
    paye_answer: str                    # Answer from PAYE agent
    final_answer: str                   # Combined final answer
    sources: list                       # Source documents
    model_used: str                     # Which LLM was used


# ============================================================================
# ROUTING LOGIC (LLM-Based - Intelligent)
# ============================================================================

async def route_query(state: AgentState) -> AgentState:
    """
    Use LLM to intelligently determine which agent(s) should handle the query.
    
    This is smarter than keyword matching - the LLM understands the semantic
    meaning of the query and routes accordingly.
    """
    query = state["query"]
    
    logger.info(f"🤔 Analyzing query for routing...")
    
    # Create routing prompt
    routing_prompt = f"""You are a routing assistant for a Nigerian tax chatbot system.

Analyze the user's query and determine which knowledge base(s) to search:

KNOWLEDGE BASES:
1. "tax" - General tax policies, VAT, corporate tax, tax laws, regulations, exemptions, reliefs, penalties
2. "paye" - PAYE (Pay As You Earn) calculations, salary tax, employee tax deductions, payroll tax
3. "both" - Queries that need information from BOTH tax policy AND PAYE calculations

USER QUERY:
{query}

INSTRUCTIONS:
- Respond with ONLY one word: "tax", "paye", or "both"
- Choose "tax" for general tax policy questions
- Choose "paye" for PAYE calculation or salary-related tax questions
- Choose "both" if the query needs information from both areas
- If unsure, choose "both"

ROUTE:"""
    
    # Use LLM to determine route
    try:
        llm_manager = LLMManager()
        llm = llm_manager.get_llm()
        
        # Invoke LLM directly (it's synchronous)
        response = llm.invoke(routing_prompt)
        
        # Extract route from response
        route = response.content.strip().lower()
        
        # Validate route
        if route not in ["tax", "paye", "both"]:
            logger.warning(f"Invalid route '{route}', defaulting to 'both'")
            route = "both"
        
        state["route"] = route
        logger.info(f"🔀 Query routed to: {route.upper()}")
        
    except Exception as e:
        logger.error(f"Error in routing: {str(e)}, defaulting to 'both'")
        state["route"] = "both"
    
    return state


# ============================================================================
# AGENT NODES
# ============================================================================

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


# ============================================================================
# Response Generator NODE (Combines answers from multiple agents)
# ============================================================================

async def response_generator(state: AgentState) -> AgentState:
    """
    Intelligently synthesize answers from multiple agents using LLM.
    This is the MAIN response generation - combines information coherently.
    """
    logger.info("🔗 Synthesizing answers with LLM...")
    
    # If we have both answers, use LLM to synthesize
    if state.get("tax_answer") and state.get("paye_answer"):
        synthesis_prompt = f"""You are a helpful Nigerian Tax Assistant. You have received information from two specialized agents about a user's tax question.

PREVIOUS CONVERSATION:
{format_chat_history(state.get("messages", []))}

USER'S ORIGINAL QUESTION:
{state['query']}

TAX POLICY INFORMATION:
{state['tax_answer']}

PAYE CALCULATION INFORMATION:
{state['paye_answer']}

INSTRUCTIONS:
Your task is to synthesize these two pieces of information into ONE coherent, comprehensive answer that directly addresses the user's question.

1. Combine the information naturally - don't just list them separately
2. Remove any redundancy between the two answers
3. Ensure the response flows logically
4. Prioritize the most relevant information for the user's question
5. Use clear, professional language
6. If there are calculations, show them clearly

Provide a single, well-structured answer that addresses the user's question completely.

SYNTHESIZED ANSWER:"""
        
        # Use LLM to synthesize
        try:
            llm_manager = LLMManager()
            llm = llm_manager.get_llm()
            response = llm.invoke(synthesis_prompt)
            
            state["final_answer"] = response.content
            state["model_used"] = llm_manager.get_active_model()
            logger.info("✅ Synthesis completed with LLM")
            
        except Exception as e:
            logger.error(f"Error in synthesis: {str(e)}, using simple combination")
            # Fallback to simple combination
            state["final_answer"] = f"""Based on Nigerian tax regulations:

TAX POLICY INFORMATION:
{state['tax_answer']}

PAYE CALCULATION DETAILS:
{state['paye_answer']}"""
    
    # If only tax answer
    elif state.get("tax_answer"):
        state["final_answer"] = state["tax_answer"]
    
    # If only PAYE answer
    elif state.get("paye_answer"):
        state["final_answer"] = state["paye_answer"]

    # Save conversation to messages
    state["messages"] = [
        HumanMessage(content=state["query"]),
        AIMessage(content=state["final_answer"])
    ]

    return state


# ============================================================================
# ROUTING LOGIC FOR GRAPH
# ============================================================================

def decide_next_step(state: AgentState) -> str:
    """
    Decide which node to execute next based on the route.
    """
    route = state.get("route", "both")
    
    if route == "tax":
        return "tax_agent"
    elif route == "paye":
        return "paye_agent"
    else:  # "both"
        return "combined_agent"


def decide_after_agents(state: AgentState) -> str:
    """
    Decide whether to finally respond or end.
    """
    # If we ran both agents separately, responder
    if state.get("tax_answer") and state.get("paye_answer"):
        return "responder"
    else:
        return "end"


# ============================================================================
# CHECKPOINTER MANAGEMENT (Global Connection Pool)
# ============================================================================

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
    
    db_url = os.getenv("DATABASE_URL")
    
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
            "combined_agent": "combined_agent"
        }
    )
    
    # Tax agent -> responder
    workflow.add_edge("tax_agent", "responder")
    
    # PAYE agent -> responder
    workflow.add_edge("paye_agent", "responder")
    
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

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def format_chat_history(messages: list) -> str:
    """Format chat history for inclusion in prompts."""
    if not messages:
        return "No previous conversation."
    
    formatted = []
    for msg in messages[-6:]:  # Only last 3 exchanges (6 messages)
        role = "user" if msg.type == "human" else "assistant"
        content = msg.content
        
        if role == "user":
            formatted.append(f"User: {content}")
        else:
            formatted.append(f"Assistant: {content}")
    
    return "\n".join(formatted)

# # ============================================================================
# MAIN QUERY FUNCTION (Simple Interface)
# ============================================================================

async def main_agent(
    query: str, 
    return_sources: bool = False,
    thread_id: str = "default"
) -> dict:
    """
    Main function to ask a tax question using the multi-agent system with memory.
    
    This is the simple interface you'll use in your application.
    
    Args:
        query: User's question
        return_sources: Whether to include source documents
        thread_id: Conversation thread ID for memory (default: "default")
    
    Returns:
        Dictionary with answer and optional sources
    
    Example:
        >>> result = await main_agent("How is PAYE calculated?")
        >>> print(result['answer'])
        
        >>> # With conversation memory
        >>> result = await main_agent("What is VAT?", thread_id="user_123")
        >>> result = await main_agent("How much is it?", thread_id="user_123")
    """
    logger.info(f"❓ Question: {query}")
    
    # Get the compiled agent (initializes once, reuses after)
    app = await get_compiled_agent()
    
    # Initial state
    initial_state = {
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

async def main():
    """Example usage of the multi-agent system."""
    
    print("="*70)
    print("MULTI-AGENT TAX SYSTEM (LangGraph)")
    print("="*70)
    
    # Example 1: Tax policy question (routes to tax agent)
    print("\n" + "="*70)
    print("EXAMPLE 1: Tax Policy Question")
    print("="*70)
    
    result1 = await main_agent(
        "What is VAT in Nigeria?",
        return_sources=True
    )
    
    print(f"\nQuestion: What is VAT in Nigeria?")
    print(f"Route: {result1['route_used']}")
    print(f"Model: {result1['model_used']}")
    print(f"\nAnswer:\n{result1['answer'][:300]}...")
    
    # Example 2: PAYE question (routes to PAYE agent)
    print("\n" + "="*70)
    print("EXAMPLE 2: PAYE Calculation Question")
    print("="*70)
    
    result2 = await main_agent(
        "How is PAYE calculated for a salary of 500,000 Naira?",
        return_sources=True
    )
    
    print(f"\nQuestion: How is PAYE calculated for 500k salary?")
    print(f"Route: {result2['route_used']}")
    print(f"Model: {result2['model_used']}")
    print(f"\nAnswer:\n{result2['answer'][:300]}...")
    
    # Example 3: Combined question (routes to both agents)
    print("\n" + "="*70)
    print("EXAMPLE 3: Combined Question")
    print("="*70)
    
    result3 = await main_agent(
        "What tax reliefs are available and how do they affect PAYE calculation?",
        return_sources=True
    )
    
    print(f"\nQuestion: Tax reliefs and PAYE?")
    print(f"Route: {result3['route_used']}")
    print(f"Model: {result3['model_used']}")
    print(f"\nAnswer:\n{result3['answer'][:300]}...")
    
    print("\n" + "="*70)
    print("✅ All examples completed!")
    print("="*70)
    
    # Cleanup: Close database connection pool
    await close_checkpointer()


if __name__ == '__main__':

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
