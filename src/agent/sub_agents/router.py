import logging
from src.agent.graph_builder.agent_state import AgentState
from src.agent.utils import format_chat_history
from src.services.llm import LLMManager

logger = logging.getLogger("router")


async def route_query(state: AgentState) -> AgentState:
    """
    LLM-powered intelligent routing with conversation context awareness.
    """
    query = state["query"]
    messages = state.get("messages", [])
    last_route = state.get("route", None)
    
    logger.info(f"🤔 Analyzing query for routing...")
    
    # Format recent conversation context
    chat_history = format_chat_history(messages[-4:]) if messages else "No previous conversation."
    
    routing_prompt = f"""You are an intelligent routing system for a Nigerian tax and financial chatbot.

CONVERSATION HISTORY:
{chat_history}

LAST ROUTE USED: {last_route if last_route else "None (first message)"}

CURRENT USER QUERY:
{query}

AVAILABLE ROUTES:
- "paye" → PAYE calculations, salary tax, employee deductions, payroll questions
- "tax" → General tax policies, VAT, corporate tax, tax laws, regulations, reliefs
- "financial" → Personal finance, investments, savings, budgeting, money management
- "both" → Queries needing BOTH tax policy AND PAYE info (not financial)

ROUTING INTELLIGENCE:
1. **Context Continuity**: If user is answering questions or providing follow-up info related to the previous route topic, STAY in that route
2. **Topic Detection**: Only switch routes when user asks about a clearly DIFFERENT topic
3. **Completion Signals**: When user signals completion ("skip", "that's all", "done") on PAYE questions, route to "paye" to finalize

Respond with ONLY ONE WORD: paye, tax, financial, or both

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
        if route not in ["tax", "paye", "both", "financial"]:
            logger.warning(f"Invalid route '{route}', defaulting to 'both'")
            route = "both"
        
        state["route"] = route      
        logger.info(f"🔀 Query routed to: {route.upper()}")
        
    except Exception as e:
        logger.error(f"Error in routing: {str(e)}, defaulting to 'both'")
        state["route"] = "both"
    
    return state