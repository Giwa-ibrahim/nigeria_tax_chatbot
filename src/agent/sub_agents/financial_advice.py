import logging
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.graph_builder.agent_state import AgentState
from src.tools.web_search import search_financial_web
from src.services.llm import LLMManager
from src.agent.utils import format_chat_history

logger = logging.getLogger("financial_advice_agent")


async def financial_advice_agent(state: AgentState) -> AgentState:
    """
    Financial Advice Agent - Handles personal finance, investment, savings questions.
    Searches Nigerian financial websites for up-to-date advice and recommendations.
    """
    logger.info("💰 Financial Advice Agent processing...")
    
    query = state["query"]
    
    # Search financial websites
    logger.info("🔍 Searching Nigerian financial websites...")
    web_results = search_financial_web(query, max_results=5)
    
    if web_results:
        # Create prompt for financial advice synthesis
        chat_history = format_chat_history(state.get("messages", []))
        history_section = ""
        if chat_history and chat_history.strip() and chat_history != "No previous conversation.":
            history_section = f"\nPREVIOUS CONVERSATION:\n{chat_history}\n"
        
        financial_prompt = f"""You are a helpful Nigerian Financial Advisor. Use the following information from trusted Nigerian financial websites to provide accurate, practical financial advice.

{history_section}

INFORMATION FROM FINANCIAL SOURCES:
{web_results}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Provide practical, actionable financial advice based on the Nigerian context
2. Be clear and direct - answer the specific question asked
3. Use information from the sources provided
4. If discussing investments, mention both opportunities and risks
5. Include specific numbers, rates, or percentages when available
6. Make recommendations relevant to the Nigerian financial landscape
7. Be conversational and relatable to young Nigerian audience
8. If the sources don't have enough information, say so clearly
9. Do NOT reference sources by number (e.g., "Source 1", "Source 2")
10. Keep the response focused and concise

FINANCIAL ADVICE:"""
        
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
