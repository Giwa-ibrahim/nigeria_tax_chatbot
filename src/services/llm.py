import logging
from typing import Optional
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from src.configurations.config import settings

# Configure logging
logger= logging.getLogger("llm")

class LLMManager:
    """
    Manages LLM instances with automatic fallback from Groq to Gemini.
    Uses Groq as the primary model and falls back to Gemini if Groq fails.
    """
    
    def __init__(self):
        """
        Initialize the LLM Manager.
        """
        self.gemini_model = settings.GEMINI_MODEL
        self.groq_model = settings.GROQ_MODEL
        self.temperature = settings.TEMPERATURE
        self.max_tokens = settings.MAX_TOKENS
        
        # Get API keys from settings
        self.gemini_api_key = settings.GOOGLE_API_KEY
        self.groq_api_key = settings.GROQ_API_KEY
        
        # Track which model is currently active
        self.active_model = None
        self.current_llm = None
        
    def _initialize_gemini(self) -> Optional[ChatGoogleGenerativeAI]:
        """Initialize Gemini LLM."""
        try:
            if not self.gemini_api_key:
                logger.warning("GOOGLE_API_KEY not found in environment variables")
                return None
            
            logger.info(f"Initializing Gemini model: {self.gemini_model}")
            
            llm = ChatGoogleGenerativeAI(
                model=self.gemini_model,
                google_api_key=self.gemini_api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
                    
            # Test the connection with a simple query
            test_response = llm.invoke("Hi")
            logger.info("✅ Gemini model initialized successfully")
            return llm
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {str(e)}")
            return None
    
    def _initialize_groq(self) -> Optional[ChatGroq]:
        """Initialize Groq LLM."""
        try:
            if not self.groq_api_key:
                logger.warning("GROQ_API_KEY not found in environment variables")
                return None
            
            logger.info(f"Initializing Groq model: {self.groq_model}")
            
            llm = ChatGroq(
                model=self.groq_model,
                groq_api_key=self.groq_api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Test the connection with a simple query
            test_response = llm.invoke("Hi")
            logger.info("✅ Groq model initialized successfully")
            return llm
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Groq: {str(e)}")
            return None
    
    def get_llm(self, force_fallback: bool = False):
        """
        Get an LLM instance with automatic fallback.
        
        Args:
            force_fallback: If True, skip Groq and use Gemini directly
            
        Returns:
            LLM instance (Groq or Gemini)
            
        Raises:
            RuntimeError: If both Groq and Gemini fail to initialize
        """
        # If we already have a working LLM, return it
        if self.current_llm and not force_fallback:
            return self.current_llm
        
        # Try Groq first (unless forced to use fallback)
        if not force_fallback:
            groq_llm = self._initialize_groq()
            if groq_llm:
                self.active_model = "groq"
                self.current_llm = groq_llm
                return groq_llm
            
            logger.warning("⚠️ Groq unavailable, falling back to Gemini...")
        
        # Fallback to Gemini
        gemini_llm = self._initialize_gemini()
        if gemini_llm:
            self.active_model = "gemini"
            self.current_llm = gemini_llm
            return gemini_llm
        
        # Both failed
        raise RuntimeError(
            "❌ Both Groq and Gemini models failed to initialize. "
        )
    
    def invoke(self, prompt: str, force_fallback: bool = False):
        """
        Invoke the LLM with automatic fallback on failure.
        
        Args:
            prompt: The prompt to send to the LLM
            force_fallback: If True, skip Gemini and use Groq directly
            
        Returns:
            LLM response
        """
        try:
            llm = self.get_llm(force_fallback=force_fallback)
            logger.info(f"Using {self.active_model.upper()} model for inference")
            response = llm.invoke(prompt)
            return response
            
        except Exception as e:
            # If Gemini fails during invocation, try Groq
            if self.active_model == "gemini" and not force_fallback:
                logger.error(f"❌ Gemini invocation failed: {str(e)}")
                logger.info("⚠️ Attempting fallback to Groq...")
                
                try:
                    return self.invoke(prompt, force_fallback=True)
                except Exception as groq_error:
                    logger.error(f"❌ Groq fallback also failed: {str(groq_error)}")
                    raise RuntimeError(f"Both models failed. Gemini: {str(e)}, Groq: {str(groq_error)}")
            else:
                raise
    
    def get_active_model(self) -> Optional[str]:
        """Get the name of the currently active model."""
        return self.active_model


# Convenience function for quick usage
def get_llm(
    gemini_model: Optional[str] = None,
    groq_model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    force_fallback: bool = False
):
    """
    Quick function to get an LLM instance with fallback support.
    
    Args:
        gemini_model: Gemini model name (default: from settings)
        groq_model: Groq model name (default: from settings)
        temperature: Temperature for response generation (default: from settings)
        max_tokens: Maximum tokens in response (default: from settings)
        force_fallback: If True, skip Gemini and use Groq directly
        
    Returns:
        LLM instance
        
    Example:
        >>> llm = get_llm()
        >>> response = llm.invoke("What is the capital of Nigeria?")
        >>> print(response.content)
    """
    manager = LLMManager(
        gemini_model=gemini_model,
        groq_model=groq_model,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return manager.get_llm(force_fallback=force_fallback)


# # Example usage
# if __name__ == "__main__":
#     # Example 1: Using LLMManager class
#     print("="*50)
#     print("Example 1: Using LLMManager")
#     print("="*50)
    
#     manager = LLMManager(temperature=0.7)
    
#     try:
#         response = manager.invoke("What is PAYE tax in Nigeria?")
#         print(f"\nActive Model: {manager.get_active_model()}")
#         print(f"Response: {response.content}\n")
#     except Exception as e:
#         print(f"Error: {str(e)}")
    
#     # Example 2: Using convenience function
#     print("="*50)
#     print("Example 2: Using convenience function")
#     print("="*50)
    
#     try:
#         llm = get_llm(temperature=0.5)
#         response = llm.invoke("Explain VAT in simple terms")
#         print(f"Response: {response.content}\n")
#     except Exception as e:
#         print(f"Error: {str(e)}")
    
#     # Example 3: Force using Gemini (fallback)
#     print("="*50)
#     print("Example 3: Forcing Gemini fallback")
#     print("="*50)
    
#     try:
#         llm = get_llm(force_fallback=True)
#         response = llm.invoke("What is tax policy?")
#         print(f"Response: {response.content}\n")
#     except Exception as e:
#         print(f"Error: {str(e)}")
