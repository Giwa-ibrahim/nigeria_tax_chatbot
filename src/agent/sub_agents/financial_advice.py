import logging
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph_builder.agent_state import AgentState
from src.tools.web_search import search_financial_web
from src.services.llm import LLMManager
from src.agent.utils import format_chat_history
from src.agent.prompt_library.system_prompts import FINANCIAL_ADVICE_PROMPT
from src.agent.prompt_library.base import get_preference_instructions

logger = logging.getLogger("financial_advice_agent")


async def financial_advice_agent(state: AgentState) -> AgentState:
    """
    Financial Advice Agent - Handles personal finance, investment, savings questions.
    Searches Nigerian financial websites for up-to-date advice and recommendations.
    """
    logger.info("💰 Financial Advice Agent processing...")
    
    query = state["query"]
    user_preferences = state.get("user_preferences", {})
    
    # Search financial websites
    logger.info("🔍 Searching Nigerian financial websites...")
    web_results = search_financial_web(query, max_results=5)
    
    if web_results:
        # Create prompt for financial advice synthesis
        chat_history = format_chat_history(state.get("messages", []))
        history_section = ""
        if chat_history and chat_history.strip() and chat_history != "No previous conversation.":
            history_section = f"\nPREVIOUS CONVERSATION:\n{chat_history}\n"
        
        pref_inst = get_preference_instructions(user_preferences)
        
        financial_prompt = FINANCIAL_ADVICE_PROMPT.format(
            history_section=history_section,
            web_results=web_results,
            query=query,
            preference_instructions=pref_inst
        )
        
        # Use LLM to generate advice
        try:
            llm_manager = LLMManager()
            llm = llm_manager.get_llm()
            response = llm.invoke(financial_prompt)
            
            state["financial_answer"] = response.content
            state["model_used"] = llm_manager.get_active_model()
            logger.info("✅ Financial advice generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating financial advice: {e}")
            state["financial_answer"] = f"I found some information from financial sources:\n\n{web_results}\n\nHowever, I encountered an issue processing this information. Please review the sources above."
            
    else:
        # No web results
        logger.warning("No financial web results found")
        state["financial_answer"] = "I couldn't find recent information from Nigerian financial websites for your question. Please try rephrasing your question or ask about specific financial topics like savings, investments, budgeting, or financial planning."
    
    # Set final_answer for the graph
    state["final_answer"] = state["financial_answer"]
    
    # Save conversation to messages
    state["messages"] = [
        HumanMessage(content=state["query"]),
        AIMessage(content=state["final_answer"])
    ]
    
    logger.info("✅ Financial Advice Agent completed")
    return state
