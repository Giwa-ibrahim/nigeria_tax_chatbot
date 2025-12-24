import logging
from src.agent.graph_builder.agent_state import AgentState
from src.services.llm import LLMManager

logger = logging.getLogger("router")


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