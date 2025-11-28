import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Import our custom modules
from llm import LLMManager
from vectors import query_vectorstore
from embeddings import get_embeddings

# Load environment variables
load_dotenv()

logger= logging.getLogger("rag")

# ============================================================================
# CORE RAG FUNCTIONS (Modular & Reusable)
# ============================================================================

def retrieve_context(
    query: str,
    collection_type: str = "both",
    top_k: int = 3,
    tax_collection: str = "tax_documents",
    paye_collection: str = "paye_calculations"
) -> List[Tuple[str, Dict, float]]:
    """
    Retrieve relevant documents from vector stores.
    
    Args:
        query: User query
        collection_type: Which collection to search ("tax", "paye", or "both")
        top_k: Number of documents to retrieve
        tax_collection: Name of the tax policy collection
        paye_collection: Name of the PAYE collection
    
    Returns:
        List of tuples: (document_text, metadata, similarity_score)
    """
    results = []
    
    try:
        # Query tax policy documents
        if collection_type in ["tax", "both"]:
            logger.info(f"Querying tax policy documents (top {top_k})...")
            tax_results = query_vectorstore(tax_collection, query, top_k=top_k)
            
            if tax_results:
                for doc, score in tax_results:
                    results.append((doc.page_content, doc.metadata, score))
                logger.info(f"Retrieved {len(tax_results)} tax policy documents")
        
        # Query PAYE documents
        if collection_type in ["paye", "both"]:
            logger.info(f"Querying PAYE documents (top {top_k})...")
            paye_results = query_vectorstore(paye_collection, query, top_k=top_k)
            
            if paye_results:
                for doc, score in paye_results:
                    results.append((doc.page_content, doc.metadata, score))
                logger.info(f"Retrieved {len(paye_results)} PAYE documents")
        
        # Sort by similarity score (lower is better for distance metrics)
        results.sort(key=lambda x: x[2])
        
        logger.info(f"Total documents retrieved: {len(results)}")
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return []


def format_context(retrieved_docs: List[Tuple[str, Dict, float]]) -> str:
    """
    Format retrieved documents into a context string for the LLM.
    
    Args:
        retrieved_docs: List of (text, metadata, score) tuples
    
    Returns:
        Formatted context string
    """
    if not retrieved_docs:
        return "No relevant documents found."
    
    context_parts = []
    for i, (text, metadata, score) in enumerate(retrieved_docs, 1):
        source = metadata.get('source', 'Unknown')
        doc_type = metadata.get('type', 'Unknown')
        
        context_parts.append(
            f"[Document {i} - {doc_type} - {source}]\n{text}\n"
        )
    
    return "\n".join(context_parts)


def create_prompt(query: str, context: str, chat_history: str) -> str:
    """
    Create a prompt for the LLM with query and context.
    
    Args:
        query: User query
        context: Retrieved context
    
    Returns:
        Formatted prompt
    """
    # Add chat history section if available
    history_section = ""
    if chat_history and chat_history.strip() and chat_history != "No previous conversation.":
        history_section = f"\nPREVIOUS CONVERSATION:\n{chat_history}\n"
    
    
    prompt = f"""You are a helpful Nigerian Tax Assistant. Use the following context from official tax documents to answer the user's question accurately and comprehensively.

{history_section}

CONTEXT:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Be specific and cite relevant tax laws, rates, or regulations when applicable
4. Use clear, professional language
5. If calculations are involved, show the steps
6. For PAYE questions, explain the calculation method clearly
7. DO NOT reference documents by number (e.g., "Document 1", "Document 2") in your response
8. Present information naturally without citing document numbers

ANSWER:"""
    
    return prompt


def generate_response(
    prompt: str,
    llm_manager: Optional[LLMManager] = None,
    force_fallback: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 1024
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
        llm_manager = LLMManager(temperature=temperature, max_tokens=max_tokens)
    
    logger.info("Generating response with LLM...")
    llm = llm_manager.get_llm(force_fallback=force_fallback)
    response = llm.invoke(prompt)
    
    # Extract answer
    answer = response.content if hasattr(response, 'content') else str(response)
    
    # Get model name
    model_used = llm_manager.get_active_model()
    
    logger.info(f"✅ Response generated successfully using {model_used}")
    
    return answer, model_used

# ============================================================================
# MAIN RAG QUERY FUNCTION (For Integration)
# ============================================================================

def query_rag(
    user_query: str,
    collection_type: str = "both",
    top_k: int = 3,
    force_fallback: bool = False,
    return_sources: bool = False,
    llm_manager: Optional[LLMManager] = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    tax_collection: str = "tax_documents",
    paye_collection: str = "paye_calculations",
    chat_history: str= " "
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

if __name__ == "__main__":
    import sys
    
    # Create a single LLM manager for all queries (more efficient)
    llm_manager = LLMManager()
    
    # Example 1: Tax Policy Query
    print("\n" + "="*60)
    print("EXAMPLE 1: Tax Policy Query")
    print("="*60)
    
    result1 = query_rag(
        user_query="What are the tax exemptions for individuals in Nigeria?",
        collection_type="tax",
        return_sources=True,
        llm_manager=llm_manager
    )
    
    print(f"\n📝 Question: What are the tax exemptions for individuals in Nigeria?")
    print(f"\n🤖 Answer ({result1['model_used']}):")
    print(result1['answer'])
    
    if result1.get('sources'):
        print(f"\n📚 Sources ({len(result1['sources'])} documents):")
        for i, source in enumerate(result1['sources'], 1):
            print(f"  {i}. {source['source']} (score: {source['score']:.4f})")
    
    # Example 2: PAYE Query
    print("\n" + "="*60)
    print("EXAMPLE 2: PAYE Query")
    print("="*60)
    
    result2 = query_rag(
        user_query="How is PAYE calculated for an employee earning 500,000 Naira monthly?",
        collection_type="paye",
        return_sources=True,
        llm_manager=llm_manager
    )
    
    print(f"\n📝 Question: How is PAYE calculated for an employee earning 500,000 Naira monthly?")
    print(f"\n🤖 Answer ({result2['model_used']}):")
    print(result2['answer'])
    
    if result2.get('sources'):
        print(f"\n📚 Sources ({len(result2['sources'])} documents):")
        for i, source in enumerate(result2['sources'], 1):
            print(f"  {i}. {source['source']} (score: {source['score']:.4f})")
    
    # Example 3: Combined Query
    print("\n" + "="*60)
    print("EXAMPLE 3: Combined Query")
    print("="*60)
    
    result3 = query_rag(
        user_query="What are the tax reliefs available and how do they affect PAYE?",
        collection_type="both",
        return_sources=True,
        force_fallback=True,
        llm_manager=llm_manager
    )
    
    print(f"\n📝 Question: What are the tax reliefs available and how do they affect PAYE?")
    print(f"\n🤖 Answer ({result3['model_used']}):")
    print(result3['answer'])
    
    if result3.get('sources'):
        print(f"\n📚 Sources ({len(result3['sources'])} documents):")
        for i, source in enumerate(result3['sources'], 1):
            print(f"  {i}. {source['source']} ({source['type']}) - score: {source['score']:.4f}")
    
    