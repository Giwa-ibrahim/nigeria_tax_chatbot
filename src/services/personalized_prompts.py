"""
Personalized Prompts Service

LLM-generated simple example prompts covering all routes.
"""

import logging, asyncio
import json
import re
from typing import List, Dict
from src.services.llm import LLMManager
from src.services.user_data import UserDataService
from src.agent.prompt_library.system_prompts import PERSONALISED_USER_PROMPT

logger = logging.getLogger("personalized_prompts")

# Cache for generated prompts: user_id -> list of prompts
_cached_prompts: Dict[str, List[str]] = {}

# Default generic prompts if no user_id is provided or generation fails
DEFAULT_PROMPTS = [
    "What is VAT in Nigeria?",
    "Calculate PAYE on ₦200,000 monthly salary",
    "Best investment options in Nigeria?",
    "How does pension contribution reduce my tax?",
    "Who is tax exempt under the 2026 policy?",
    "What deductions can reduce my PAYE?",
    "How to save money while paying taxes?",
    "Explain the new tax reform and its impact on salaries"
]

async def get_personalized_prompts(user_id: str = None) -> List[str]:
    """
    Generate 8 diverse example prompts using LLM based on user profile (cached).
    
    Args:
        user_id: The ID of the user to personalize prompts for.
        
    Returns:
        List of 8 question strings covering tax, PAYE, financial, and combined topics.
    """
    global _cached_prompts
    
    if not user_id:
        return DEFAULT_PROMPTS
        
    # Check cache
    if user_id in _cached_prompts:
        return _cached_prompts[user_id]
        
    # Fetch user data to personalize prompts
    try:
        user_profile = await UserDataService.get_user_profile(user_id)
    except Exception as e:
        logger.warning(f"Could not load profile for prompts: {e}")
        user_profile = {}
        
    # Determine context to inject
    user_context = "The user is a general user in Nigeria."
    if user_profile:
        salary = user_profile.get("monthly_salary") or user_profile.get("annual_salary")
        if salary:
            user_context = f"The user earns {salary}. Please suggest realistic questions tailored to this income level."
        
    generation_prompt = PERSONALISED_USER_PROMPT.format(user_context=user_context)
    
    try:
        llm_manager = LLMManager()
        llm = llm_manager.get_llm()
        # Use asyncio.to_thread because LLMManager implements invoke(), not ainvoke()
       
        response = await asyncio.to_thread(llm.invoke, generation_prompt)
        
        # Extract JSON array from response
        content = response.content.strip()
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        prompts = json.loads(content)
        
        if not isinstance(prompts, list) or len(prompts) < 8:
            logger.warning(f"LLM generated {len(prompts)} prompts, expected 8")
            if len(prompts) < 8:
                prompts.extend(DEFAULT_PROMPTS)
        
        logger.info(f"✅ Generated {len(prompts)} personalized prompts for user {user_id}")
        _cached_prompts[user_id] = prompts[:8]
        return _cached_prompts[user_id]
        
    except Exception as e:
        logger.error(f"Error generating prompts for user {user_id}: {e}")
        return DEFAULT_PROMPTS
