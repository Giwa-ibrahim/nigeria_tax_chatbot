import logging
import cohere
from typing import List, Dict, Tuple, Optional
from src.vector_db.vectors import query_vectorstore
from src.configurations.config import settings

logger = logging.getLogger("doc_retriever")


def rerank_with_cohere(
    query: str,
    documents: List[Tuple[str, Dict, float]],
    top_k: int = 3
) -> List[Tuple[str, Dict, float]]:
    """
    Rerank retrieved documents using Cohere's reranking API.
    Filters out irrelevant results and keeps only the most relevant ones.
    
    Args:
        query: User's question
        documents: List of (text, metadata, score) tuples
        top_k: Number of top documents to keep after reranking
    
    Returns:
        Reranked and filtered list of documents
    """
    if not documents:
        return []
    
    try:
        # Initialize Cohere client
        co = cohere.Client(settings.COHERE_API_KEY)
        
        # Extract just the text for reranking
        doc_texts = [doc[0] for doc in documents]
        
        logger.info(f"🔄 Reranking {len(doc_texts)} documents with Cohere...")
        
        # Rerank using Cohere
        rerank_results = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=doc_texts,
            top_n=top_k,
            return_documents=True
        )
        
        # Rebuild document list with reranked order and scores
        reranked_docs = []
        for result in rerank_results.results:
            original_doc = documents[result.index]
            # Use Cohere's relevance score (0-1, higher is better)
            reranked_docs.append((
                original_doc[0],  # text
                original_doc[1],  # metadata
                result.relevance_score  # Cohere relevance score
            ))
        
        logger.info(f"✅ Reranking complete - kept top {len(reranked_docs)} relevant docs")
        
        # Log relevance scores for debugging
        for i, (_, _, score) in enumerate(reranked_docs, 1):
            logger.info(f"  Doc {i}: Relevance score = {score:.3f}")
        
        return reranked_docs
        
    except ImportError:
        logger.warning("⚠️ Cohere not installed - install with: pip install cohere")
        return documents[:top_k]
    except Exception as e:
        logger.error(f"❌ Error in Cohere reranking: {str(e)}")
        # Fallback to original results
        return documents[:top_k]


def retrieve_context(
    query: str,
    collection_type: str = "both",
    top_k: int = 3,
    tax_collection: str = "tax_documents",
    paye_collection: str = "paye_calculations",
    use_reranking: bool = True
) -> List[Tuple[str, Dict, float]]:
    """
    Retrieve relevant documents from vector stores with optional Cohere reranking.
    
    Args:
        query: User query
        collection_type: Which collection to search ("tax", "paye", or "both")
        top_k: Number of documents to retrieve after reranking
        tax_collection: Name of the tax policy collection
        paye_collection: Name of the PAYE collection
        use_reranking: Whether to use Cohere reranking (default: True)
    
    Returns:
        List of tuples: (document_text, metadata, relevance_score)
    """
    results = []
    
    # Retrieve more documents initially for reranking (2x top_k)
    initial_k = top_k * 2 if use_reranking else top_k
    
    try:
        # Query tax policy documents
        if collection_type in ["tax", "both"]:
            logger.info(f"Querying tax policy documents (top {initial_k})...")
            tax_results = query_vectorstore(tax_collection, query, top_k=initial_k)
            
            if tax_results:
                for doc, score in tax_results:
                    results.append((doc.page_content, doc.metadata, score))
                logger.info(f"Retrieved {len(tax_results)} tax policy documents")
        
        # Query PAYE documents
        if collection_type in ["paye", "both"]:
            logger.info(f"Querying PAYE documents (top {initial_k})...")
            paye_results = query_vectorstore(paye_collection, query, top_k=initial_k)
            
            if paye_results:
                for doc, score in paye_results:
                    results.append((doc.page_content, doc.metadata, score))
                logger.info(f"Retrieved {len(paye_results)} PAYE documents")
        
        if not results:
            logger.warning("No documents retrieved")
            return []
        
        # Apply Cohere reranking to filter out irrelevant results
        if use_reranking and len(results) > 0:
            results = rerank_with_cohere(query, results, top_k=top_k)
        else:
            # Sort by similarity score and limit
            results.sort(key=lambda x: x[2])
            results = results[:top_k]
        
        logger.info(f"✅ Final documents to use: {len(results)}")
        return results
        
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}")
        return []
