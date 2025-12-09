import logging
from typing import Optional, Tuple
from src.services.llm import LLMManager
from src.configurations.config import settings
logger = logging.getLogger('rag_generator')


def generate_response(
    prompt: str,
    llm_manager: Optional[LLMManager] = None,
    force_fallback: bool = False,
    temperature: float = settings.TEMPERATURE,
    max_tokens: int = settings.MAX_TOKENS
) -> Tuple[str, str]:
    """
    Generate a response using LLM.
    
    Args:
        prompt: The prompt to send to the LLM
        llm_manager: Optional LLMManager instance (creates new one if not provided)
        force_fallback: Force use of Groq instead of Gemini
        temperature: LLM temperature for response generation
        max_tokens: Maximum tokens in LLM response
    
    Returns:
        Tuple of (answer, model_used)
    """
    # Create LLM manager if not provided
    if llm_manager is None:
        llm_manager = LLMManager()
        llm_manager.temperature = temperature
        llm_manager.max_tokens = max_tokens
    
    logger.info("Generating response with LLM...")
    llm = llm_manager.get_llm(force_fallback=force_fallback)
    response = llm.invoke(prompt)
    
    # Extract answer
    answer = response.content if hasattr(response, 'content') else str(response)
    
    # Get model name
    model_used = llm_manager.get_active_model()
    
    logger.info(f"✅ Response generated successfully using {model_used}")
    
    return answer, model_used
