"""
Personalized Prompts Service

LLM-generated simple example prompts covering all routes.
"""

import logging
import json
import re
from typing import List
from src.services.llm import LLMManager

logger = logging.getLogger("personalized_prompts")

# Cache for generated prompts
_cached_prompts = None


def get_personalized_prompts() -> List[str]:
    """
    Generate 8 diverse example prompts using LLM (cached).
    
    Returns:
        List of 8 question strings covering tax, PAYE, financial, and combined topics.
    """
    global _cached_prompts
    
    if _cached_prompts is not None:
        return _cached_prompts
    
    generation_prompt = """Generate 8 diverse example questions for a Nigerian tax and financial chatbot.

REQUIREMENTS:
1. Cover ALL 4 capabilities:
   - Tax policies (VAT, corporate tax, exemptions, reliefs)
   - PAYE calculations (salary tax, deductions)
   - Financial advice (investments, savings, budgeting)
   - Combined (tax + PAYE interactions)

2. Practical and relatable to everyday Nigerians
3. Use realistic amounts (₦150k, ₦300k, ₦500k monthly salaries)
4. Mix simple and complex questions
5. Include Nigeria Tax Act 2025/2026 topics

EXAMPLES:
- "What is VAT and how much do I pay in Nigeria?"
- "Calculate my PAYE on ₦250,000 monthly salary"
- "Best investment options for someone earning ₦300k/month"
- "How does pension contribution reduce my tax?"

Respond with ONLY a JSON array of 8 question strings:
["question 1?", "question 2?", ...]

JSON RESPONSE:"""
    
    try:
        llm_manager = LLMManager()
        llm = llm_manager.get_llm()
        response = llm.invoke(generation_prompt)
        
        # Extract JSON array from response
        content = response.content.strip()
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        prompts = json.loads(content)
        
        if not isinstance(prompts, list) or len(prompts) < 8:
            logger.warning(f"LLM generated {len(prompts)} prompts, expected 8")
        
        logger.info(f"✅ Generated {len(prompts)} personalized prompts")
        _cached_prompts = prompts[:8]
        return _cached_prompts
        
    except Exception as e:
        logger.error(f"Error generating prompts: {e}")
        # Fallback
        _cached_prompts = [
            "What is VAT in Nigeria?",
            "Calculate PAYE on ₦200,000 monthly salary",
            "Best investment options in Nigeria?",
            "How does pension contribution reduce my tax?",
            "Who is tax exempt under the 2026 policy?",
            "What deductions can reduce my PAYE?",
            "How to save money while paying taxes?",
            "Explain the new tax reform and its impact on salaries"
        ]
        return _cached_prompts
