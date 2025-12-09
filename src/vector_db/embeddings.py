from langchain_huggingface import HuggingFaceEmbeddings
import logging
from src.configurations.config import settings

logger = logging.getLogger("embeddings")

def get_embeddings():
    """
    Get HuggingFace embedding model.
    Uses sentence-transformers - FREE, no API key needed, runs locally in your app.
    Perfect for deployment - no separate server required!
    """
    try:
        model_name = settings.HUGGINGFACE_EMBED_MODEL
        logger.info(f"Initializing HuggingFace embeddings ({model_name})...")
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use CPU (works everywhere)
            encode_kwargs={'normalize_embeddings': True}  # Better similarity search
        )
        
        # Test the embeddings
        test_embed = embeddings.embed_query("test")
        logger.info("✅ HuggingFace embeddings initialized successfully")
        logger.info(f"   Model: {model_name}")
        logger.info(f"   Embedding dimension: {len(test_embed)}")
        
        return embeddings
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize HuggingFace embeddings: {str(e)}")
        raise RuntimeError(
            f"Failed to initialize embeddings. Error: {str(e)}\n"
            "Make sure 'sentence-transformers' is installed: pip install sentence-transformers"
        )
