from typing import Dict, Optional

def get_preference_instructions(user_preferences: Optional[Dict]) -> str:
    """
    Generate an instruction string based on learned user preferences.
    To be injected into LLM prompts.
    """
    if not user_preferences:
        return "Use a clear, balanced, and professional communication style."
        
    style = user_preferences.get("communication_style", "balanced")
    
    if style == "concise":
        return "INSTRUCTION: The user prefers a highly CONCISE and direct communication style. Do not use filler words. Get straight to the point."
    elif style == "detailed":
        return "INSTRUCTION: The user prefers a DETAILED communication style. Provide thorough explanations, breakdowns, and context."
    else:
        return "INSTRUCTION: Use a clear, balanced, and professional communication style."
