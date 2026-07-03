import logging
import json
import re
from src.agent.graph_builder.agent_state import AgentState
from src.agent.utils import format_chat_history
from src.services.llm import LLMManager
from src.agent.prompt_library.system_prompts import ROUTING_PROMPT

logger = logging.getLogger("router")


async def route_query(state: AgentState) -> AgentState:
    """
    Combined router + meta-analysis in a single LLM call
    """
    query = state["query"]
    messages = state.get("messages", [])

    logger.info("🤔 Analyzing query for routing + meta-analysis...")

    chat_history = format_chat_history(messages[-4:]) if messages else "No previous conversation."

    routing_prompt = ROUTING_PROMPT.format(
        chat_history=chat_history,
        query=query
    )

    try:
        llm_manager = LLMManager(model_tier="fast")  
        llm = llm_manager.get_llm()
        response = llm.invoke(routing_prompt)

        content = response.content.strip()
        # Extract JSON block in case model wraps it in markdown
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)

        result = json.loads(content)

        route = result.get("route", "both").lower()
        if route not in ["tax", "paye", "both", "financial", "general"]:
            logger.warning(f"Invalid route '{route}', defaulting to 'both'")
            route = "both"

        result["route"] = route
        state["route"] = route
        state["meta_analysis"] = result

        # Handle greetings/chitchat inline - no need for a separate agent node
        if route == "general":
            state["final_answer"] = (
                "Hello! I'm your Nigerian Tax and Financial Assistant. "
                "I can help with PAYE calculations, tax policies, or financial advice. What's your question?"
            )
            logger.info("👋 General query handled inline (greeting/chitchat)")
        else:
            logger.info(f"🔀 Query routed to: {route.upper()} | user_ctx: {result.get('needs_user_context')} | calc: {result.get('is_calculation_request')}")

    except Exception as e:
        logger.error(f"Error in routing: {e}, defaulting to 'both'")
        state["route"] = "both"
        state["meta_analysis"] = {
            "route": "both",
            "needs_user_context": False,
            "is_calculation_request": False,
            "needs_clarification": False,
            "missing_info": [],
            "user_mood": "neutral",
            "approach": "direct"
        }

    return state
