import logging
from typing import Dict, Optional
from dotenv import load_dotenv

# Import from modular components
from src.tools.retrieval.retriever import retrieve_context
from src.tools.retrieval.formatter import format_context, create_prompt
from src.tools.retrieval.generator import generate_response
from src.services.llm import LLMManager
from src.configurations.config import settings
# Load environment variables
load_dotenv()

logger = logging.getLogger("rag")


# ============================================================================
# MAIN RAG QUERY FUNCTION
# ============================================================================

def query_rag(
    user_query: str,
    collection_type: str = "both",
    top_k: int = 3,
    force_fallback: bool = False,
    return_sources: bool = False,
    llm_manager: Optional[LLMManager] = None,
    temperature: float = settings.TEMPERATURE,
    max_tokens: int = settings.MAX_TOKENS,
    tax_collection: str = "tax_documents",
    paye_collection: str = "paye_calculations",
    chat_history: str = " "
) -> Dict:
    """
    Main RAG pipeline query function - MODULAR & REUSABLE.
    
    This is the primary function to use for integrating RAG into your agentic system.
    
    Args:
        user_query: User's question
        collection_type: Which collection to search ("tax", "paye", or "both")
        top_k: Number of documents to retrieve
        force_fallback: Force use of Groq instead of Gemini
        return_sources: Include source documents in response
        llm_manager: Optional LLMManager instance (reuse for efficiency)
        temperature: LLM temperature for response generation
        max_tokens: Maximum tokens in LLM response
        tax_collection: Name of the tax policy collection
        paye_collection: Name of the PAYE collection
    
    Returns:
        Dictionary containing:
            - answer: Generated response
            - sources: List of source documents (if return_sources=True)
            - model_used: Name of the LLM model used
    
    Example:
        >>> result = query_rag("What is PAYE?", collection_type="paye")
        >>> print(result['answer'])
    """
    logger.info(f"Processing query: '{user_query[:100]}...'")
    
    try:
        # Step 1: Retrieve relevant documents
        retrieved_docs = retrieve_context(
            query=user_query,
            collection_type=collection_type,
            top_k=top_k,
            tax_collection=tax_collection,
            paye_collection=paye_collection
        )
        
        if not retrieved_docs:
            logger.warning("No relevant documents found")
            return {
                "answer": "I couldn't find relevant information in the tax documents to answer your question. Please try rephrasing or ask about Nigerian tax policies, PAYE calculations, or tax regulations.",
                "sources": [],
                "model_used": None
            }
        
        # Step 2: Format context
        context = format_context(retrieved_docs)
        
        # Step 3: Create prompt
        prompt = create_prompt(user_query, context, chat_history)
        
        # Step 4: Generate response using LLM
        answer, model_used = generate_response(
            prompt=prompt,
            llm_manager=llm_manager,
            force_fallback=force_fallback,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Prepare result
        result = {
            "answer": answer,
            "model_used": model_used
        }
        
        # Include sources if requested
        if return_sources:
            sources = [
                {
                    "text": text[:200] + "..." if len(text) > 200 else text,
                    "source": metadata.get('source'),
                    "type": metadata.get('type'),
                    "score": float(score)
                }
                for text, metadata, score in retrieved_docs
            ]
            result["sources"] = sources
        
        return result
        
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        return {
            "answer": f"An error occurred while processing your query: {str(e)}",
            "sources": [],
            "model_used": None
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

# if __name__ == "__main__":
#     import sys
    
#     # Create a single LLM manager for all queries (more efficient)
#     llm_manager = LLMManager()
    
#     # Example 1: Tax Policy Query
#     print("\n" + "="*60)
#     print("EXAMPLE 1: Tax Policy Query")
#     print("="*60)
    
#     result1 = query_rag(
#         user_query="What are the tax exemptions for individuals in Nigeria?",
#         collection_type="tax",
#         return_sources=True,
#         llm_manager=llm_manager
#     )
    
#     print(f"\n📝 Question: What are the tax exemptions for individuals in Nigeria?")
#     print(f"\n🤖 Answer ({result1['model_used']}):")
#     print(result1['answer'])
    
#     if result1.get('sources'):
#         print(f"\n📚 Sources ({len(result1['sources'])} documents):")
#         for i, source in enumerate(result1['sources'], 1):
#             print(f"  {i}. {source['source']} (score: {source['score']:.4f})")
    
#     # Example 2: PAYE Query
#     print("\n" + "="*60)
#     print("EXAMPLE 2: PAYE Query")
#     print("="*60)
    
#     result2 = query_rag(
#         user_query="How is PAYE calculated for an employee earning 500,000 Naira monthly?",
#         collection_type="paye",
#         return_sources=True,
#         llm_manager=llm_manager
#     )
    
#     print(f"\n📝 Question: How is PAYE calculated for an employee earning 500,000 Naira monthly?")
#     print(f"\n🤖 Answer ({result2['model_used']}):")
#     print(result2['answer'])
    
#     if result2.get('sources'):
#         print(f"\n📚 Sources ({len(result2['sources'])} documents):")
#         for i, source in enumerate(result2['sources'], 1):
#             print(f"  {i}. {source['source']} (score: {source['score']:.4f})")
    
#     # Example 3: Combined Query
#     print("\n" + "="*60)
#     print("EXAMPLE 3: Combined Query")
#     print("="*60)
    
#     result3 = query_rag(
#         user_query="What are the tax reliefs available and how do they affect PAYE?",
#         collection_type="both",
#         return_sources=True,
#         force_fallback=True,
#         llm_manager=llm_manager
#     )
    
#     print(f"\n📝 Question: What are the tax reliefs available and how do they affect PAYE?")
#     print(f"\n🤖 Answer ({result3['model_used']}):")
#     print(result3['answer'])
    
#     if result3.get('sources'):
#         print(f"\n📚 Sources ({len(result3['sources'])} documents):")
#         for i, source in enumerate(result3['sources'], 1):
#             print(f"  {i}. {source['source']} ({source['type']}) - score: {source['score']:.4f}")