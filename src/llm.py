import os
import logging
from typing import Optional
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class LLMManager:
    """
    Manages LLM instances with automatic fallback from Gemini to Groq.
    Uses Gemini (free) as the primary model and falls back to Groq if Gemini fails.
    """
    
    def __init__(
        self,
        gemini_model: Optional[str] = None,
        groq_model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize the LLM Manager.
        
        Args:
            gemini_model: Gemini model name
            groq_model: Groq model name
            temperature: Temperature for response generation
            max_tokens: Maximum tokens in response
        """
        # Read from environment variables with fallbacks
        self.gemini_model = os.getenv("GEMINI_MODEL")
        self.groq_model = os.getenv("GROQ_MODEL")
        self.temperature = float(os.getenv("TEMPERATURE"))
        self.max_tokens = int(os.getenv("MAX_TOKENS"))
        
        # Get API keys from environment
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Track which model is currently active
        self.active_model = None
        self.current_llm = None
        
        # Log configuration
        logging.info(f"LLM Configuration: Gemini={self.gemini_model}, Groq={self.groq_model}, Temperature={self.temperature}")
        
    def _initialize_gemini(self) -> Optional[ChatGoogleGenerativeAI]:
        """Initialize Gemini LLM."""
        try:
            if not self.gemini_api_key:
                logging.warning("GOOGLE_API_KEY not found in environment variables")
                return None
            
            logging.info(f"Initializing Gemini model: {self.gemini_model}")
            
            llm = ChatGoogleGenerativeAI(
                model=self.gemini_model,
                google_api_key=self.gemini_api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
                    
            # Test the connection with a simple query
            test_response = llm.invoke("Hi")
            logging.info("✅ Gemini model initialized successfully")
            return llm
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize Gemini: {str(e)}")
            return None
    
    def _initialize_groq(self) -> Optional[ChatGroq]:
        """Initialize Groq LLM as fallback."""
        try:
            if not self.groq_api_key:
                logging.warning("GROQ_API_KEY not found in environment variables")
                return None
            
            logging.info(f"Initializing Groq model: {self.groq_model}")
            
            llm = ChatGroq(
                model=self.groq_model,
                groq_api_key=self.groq_api_key,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Test the connection with a simple query
            test_response = llm.invoke("Hi")
            logging.info("✅ Groq model initialized successfully")
            return llm
            
        except Exception as e:
            logging.error(f"❌ Failed to initialize Groq: {str(e)}")
            return None
    
    def get_llm(self, force_fallback: bool = False):
        """
        Get an LLM instance with automatic fallback.
        
        Args:
            force_fallback: If True, skip Gemini and use Groq directly
            
        Returns:
            LLM instance (Gemini or Groq)
            
        Raises:
            RuntimeError: If both Gemini and Groq fail to initialize
        """
        # If we already have a working LLM, return it
        if self.current_llm and not force_fallback:
            return self.current_llm
        
        # Try Gemini first (unless forced to use fallback)
        if not force_fallback:
            gemini_llm = self._initialize_gemini()
            if gemini_llm:
                self.active_model = "gemini"
                self.current_llm = gemini_llm
                return gemini_llm
            
            logging.warning("⚠️ Gemini unavailable, falling back to Groq...")
        
        # Fallback to Groq
        groq_llm = self._initialize_groq()
        if groq_llm:
            self.active_model = "groq"
            self.current_llm = groq_llm
            return groq_llm
        
        # Both failed
        raise RuntimeError(
            "❌ Both Gemini and Groq models failed to initialize. "
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
            logging.info(f"Using {self.active_model.upper()} model for inference")
            response = llm.invoke(prompt)
            return response
            
        except Exception as e:
            # If Gemini fails during invocation, try Groq
            if self.active_model == "gemini" and not force_fallback:
                logging.error(f"❌ Gemini invocation failed: {str(e)}")
                logging.info("⚠️ Attempting fallback to Groq...")
                
                try:
                    return self.invoke(prompt, force_fallback=True)
                except Exception as groq_error:
                    logging.error(f"❌ Groq fallback also failed: {str(groq_error)}")
                    raise RuntimeError(f"Both models failed. Gemini: {str(e)}, Groq: {str(groq_error)}")
            else:
                raise
    
    def get_active_model(self) -> Optional[str]:
        """Get the name of the currently active model."""
        return self.active_model


# Convenience function for quick usage
def get_llm(
    gemini_model: Optional[str] = os.getenv("GEMINI_MODEL"),
    groq_model: Optional[str] = os.getenv("GROQ_MODEL"),
    temperature: Optional[float] = os.getenv("TEMPERATURE"),
    max_tokens: Optional[int] = os.getenv("MAX_TOKENS"),
    force_fallback: bool = False
):
    """
    Quick function to get an LLM instance with fallback support.
    
    Args:
        gemini_model: Gemini model name
        groq_model: Groq model name (default: from .env or llama-3.1-70b-versatile)
        temperature: Temperature for response generation (default: from .env or 0.7)
        max_tokens: Maximum tokens in response
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


# Example usage
if __name__ == "__main__":
    # Example 1: Using LLMManager class
    print("="*50)
    print("Example 1: Using LLMManager")
    print("="*50)
    
    manager = LLMManager(temperature=0.7)
    
    try:
        response = manager.invoke("What is PAYE tax in Nigeria?")
        print(f"\nActive Model: {manager.get_active_model()}")
        print(f"Response: {response.content}\n")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Example 2: Using convenience function
    print("="*50)
    print("Example 2: Using convenience function")
    print("="*50)
    
    try:
        llm = get_llm(temperature=0.5)
        response = llm.invoke("Explain VAT in simple terms")
        print(f"Response: {response.content}\n")
    except Exception as e:
        print(f"Error: {str(e)}")
    
    # Example 3: Force using Groq (fallback)
    print("="*50)
    print("Example 3: Forcing Groq fallback")
    print("="*50)
    
    try:
        llm = get_llm(force_fallback=True)
        response = llm.invoke("What is tax policy?")
        print(f"Response: {response.content}\n")
    except Exception as e:
        print(f"Error: {str(e)}")
