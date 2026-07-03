import logging
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph_builder.agent_state import AgentState
from src.services.llm import LLMManager
from src.agent.utils import format_chat_history
from src.agent.prompt_library.system_prompts import RESPONSE_SYNTHESIS_PROMPT

logger = logging.getLogger("response_generator")


async def response_generator(state: AgentState) -> AgentState:
    """
    Intelligently synthesize answers from multiple agents using LLM.
    This is the MAIN response generation - combines information coherently.
    Works for tax, PAYE, and any combination of specialized information.
    """
    logger.info("🔗 Synthesizing answers with LLM...")
    
    # If we have both tax and PAYE answers, synthesize them
    if state.get("tax_answer") and state.get("paye_answer"):
        synthesis_prompt = RESPONSE_SYNTHESIS_PROMPT.format(
            chat_history=format_chat_history(state.get("messages", [])),
            query=state['query'],
            tax_answer=state['tax_answer'],
            paye_answer=state['paye_answer']
        )
        
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
            state["final_answer"] = f"""Based on available information:

TAX POLICY:
{state['tax_answer']}

PAYE DETAILS:
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


def decide_next_step(state: AgentState) -> str:
    """
    Decide which node to execute next based on the route.
    """
    route = state.get("route", "both")
    
    if route == "tax":
        return "tax_agent"
    elif route == "paye":
        return "paye_agent"
    elif route == "financial":
        return "financial_agent"
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
