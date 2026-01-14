from langchain_cohere import CohereEmbeddings
from src.configurations.config import settings

def get_embeddings():
    """
    Get Embeddings - Optimized for Cohere rate limits
    Using light model to reduce token usage and cost
    """
    embeddings = CohereEmbeddings(
        model="embed-english-light-v3.0", 
        cohere_api_key=settings.COHERE_API_KEY,
        max_retries=5,  
        request_timeout=120  
    )
    
    return embeddings