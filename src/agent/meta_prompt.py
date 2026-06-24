import logging, json, re
from typing import Dict, Optional
from src.services.llm import LLMManager
from src.agent.utils import format_chat_history
from src.agent.prompt_library.meta_prompts import (
    META_ANALYSIS_PROMPT,
    CLARIFICATION_PROMPT,
    CONDITIONAL_PROMPT,
    ENGAGEMENT_PROMPT
)
from src.agent.prompt_library.base import get_preference_instructions

logger = logging.getLogger("meta_prompt")


def analyze_query_for_tax_calculation(query: str, chat_history: str = "") -> Dict:
    """
    LLM-powered meta-analysis: Intelligently determines what's needed for tax calculations
    and the best interaction strategy.
    """
    history_str = chat_history if chat_history else "No previous conversation"
    meta_analysis_prompt = META_ANALYSIS_PROMPT.format(
        chat_history=history_str,
        query=query
    )

    try:
        llm_manager = LLMManager()
        llm = llm_manager.get_llm()
        response = llm.invoke(meta_analysis_prompt)

        # Extract JSON from response (in case LLM adds extra text)
        content = response.content.strip()
        # Try to find JSON block
        json_match = re.search(r'\{[^}]+\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        analysis = json.loads(content)
        logger.info(f"✅ Meta-analysis: {analysis['approach']} approach, Calc request: {analysis.get('is_calculation_request', False)}")
        return analysis
        
    except Exception as e:
        logger.error(f"Error in meta-analysis: {e}")
        # fallback: Basic parsing to detect salary presence
        has_salary = bool(re.search(r'\d+[k\s]|₦\s*\d+|\d+,\d+', query.lower()))
        
        return {
            "is_calculation_request": has_salary,
            "needs_clarification": has_salary,
            "missing_info": ["pension", "nhf", "rent"] if has_salary else [],
            "user_mood": "neutral",
            "approach": "collect" if has_salary else "direct",
            "reasoning": "Fallback due to parsing error"
        }


def generate_clarification_request(missing_info: list, user_mood: str, user_query: str = "", user_preferences: Optional[Dict] = None) -> str:
    """
    LLM generates personalized, friendly clarification requests.
    """
    if user_mood == "impatient":
        return None
    
    missing_info_str = ', '.join(missing_info)
    pref_inst = get_preference_instructions(user_preferences)
    
    clarification_prompt = CLARIFICATION_PROMPT.format(
        user_query=user_query,
        user_mood=user_mood,
        missing_info_str=missing_info_str,
        preference_instructions=pref_inst
    )
    
    try:
        llm_manager = LLMManager()
        llm = llm_manager.get_llm()
        response = llm.invoke(clarification_prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating clarification: {e}")
        return None


def generate_conditional_answer(query: str, missing_info: list, partial_answer: str, user_preferences: Optional[Dict] = None) -> str:
    """
    LLM generates conditional answer with examples when user is impatient.
    """
    missing_info_str = ', '.join(missing_info)
    pref_inst = get_preference_instructions(user_preferences)
    
    conditional_prompt = CONDITIONAL_PROMPT.format(
        query=query,
        missing_info_str=missing_info_str,
        partial_answer=partial_answer,
        preference_instructions=pref_inst
    )

    try:
        llm_manager = LLMManager()
        llm = llm_manager.get_llm()
        response = llm.invoke(conditional_prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating conditional answer: {e}")
        return partial_answer


def create_engagement_response(query: str, context: str, chat_history: str = "", user_preferences: Optional[Dict] = None) -> str:
    """
    LLM creates engaging educational response for interested users.
    """
    history_str = chat_history if chat_history else "First message"
    pref_inst = get_preference_instructions(user_preferences)
    
    engagement_prompt = ENGAGEMENT_PROMPT.format(
        chat_history=history_str,
        query=query,
        context=context,
        preference_instructions=pref_inst
    )

    try:
        llm_manager = LLMManager()
        llm = llm_manager.get_llm()
        response = llm.invoke(engagement_prompt)
        return response.content
    except Exception as e:
        logger.error(f"Error generating engagement response: {e}")
        return context
