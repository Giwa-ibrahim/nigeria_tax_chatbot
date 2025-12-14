import logging
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph_builder.agent_state import AgentState
from src.services.llm import LLMManager
from src.agent.utils import format_chat_history

logger = logging.getLogger("response_generator")


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
5. Make the response contaxtualized to a Nigerian audience
6. Make the response relatable, friendly, slightly funny, contain humour and with Nigerian nuances, without compromising on the facts
7. Also make the response slightly tailored to a Nigerian younger population (between 18 - 45 years), providing relatable responses, recommendations and suggestions
8. If there are calculations, show them clearly

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
