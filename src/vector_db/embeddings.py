from langchain_openai import OpenAIEmbeddings
from src.configurations.config import settings

def get_embeddings():
    """
    Get Embeddings
    """
    
    embeddings = OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_NAME,
        openai_api_key=settings.OPENAI_API_KEY
    )
    
    return embeddings