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
Your task is to synthesize these two pieces of information into ONE concise, clear answer that directly addresses the user's question.

1. Be CONCISE and DIRECT - answer the specific question asked
2. Combine information naturally - avoid listing them separately
3. Remove ALL redundancy between the two answers
4. Prioritize the most relevant information for the user's question
5. Use bullet points or short paragraphs for clarity
6. Make the response relatable and friendly with Nigerian nuances
7. Add a touch of humor where appropriate (without compromising facts)
8. Tailor to a Nigerian younger audience (18-45 years) when relevant
9. If there are calculations, show only the essential steps clearly
10. Keep the overall response focused and to-the-point

Provide a single, well-structured answer that addresses the user's question completely WITHOUT unnecessary elaboration.

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
