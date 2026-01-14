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
    Works for tax, PAYE, and any combination of specialized information.
    """
    logger.info("🔗 Synthesizing answers with LLM...")
    
    # If we have both tax and PAYE answers, synthesize them
    if state.get("tax_answer") and state.get("paye_answer"):
        synthesis_prompt = f"""You are a knowledgeable, friendly Nigerian assistant who helps with tax and financial matters. You have received information from specialized knowledge sources.

PREVIOUS CONVERSATION:
{format_chat_history(state.get("messages", []))}

USER'S ORIGINAL QUESTION:
{state['query']}

TAX POLICY INFORMATION:
{state['tax_answer']}

PAYE CALCULATION INFORMATION:
{state['paye_answer']}

INSTRUCTIONS FOR YOUR RESPONSE:
Your primary goal is to deliver a clear, accurate, and helpful answer that directly addresses what the user asked.

LANGUAGE ADAPTATION:
0. **CRITICAL**: Detect the language the user is using in their question
   - If user speaks in Nigerian Pidgin English, respond ENTIRELY in Pidgin (e.g., "Wetin be...", "E dey...", "Na so...", "Oga/Sister...")
   - If user speaks in Standard English, respond in Standard English
   - If user mixes both, use a light Pidgin-influenced Nigerian English
   - Mirror their communication style naturally

TONE & STYLE:
1. Be conversational and relatable - speak like a knowledgeable friend, not a robot
2. Use Nigerian context and nuances where relevant (e.g., "Naira" not just "₦")
3. Add personality - a touch of warmth or light humor when appropriate (but never at the expense of accuracy)
4. Tailor your language to a young, educated Nigerian audience (18-45 years)
5. Be encouraging and empowering - help users feel confident about their financial decisions

CONTENT STRUCTURE:
6. Start with a DIRECT answer to their specific question
7. Synthesize both information sources into ONE cohesive response (not separate sections)
8. Remove ALL redundancy - say things once, clearly
9. Use bullet points, numbering, or short paragraphs for readability
10. If there are calculations, show essential steps with clear explanations
11. Highlight actionable takeaways when relevant

ACCURACY & RELEVANCE:
12. Base everything on the information provided - no hallucinations
13. Prioritize the most relevant details for the user's specific question
14. If citing specific rates, laws, or dates, be precise
15. If the information is insufficient, acknowledge it honestly

BREVITY:
16. Be comprehensive but concise - quality over quantity
17. Avoid unnecessary elaboration or tangents
18. Don't repeat information already clear in the context

Provide your synthesized answer now:"""
        
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
