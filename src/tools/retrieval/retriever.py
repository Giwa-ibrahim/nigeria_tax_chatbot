import logging
from typing import List, Dict, Tuple
from src.vector_db.vectors import query_vectorstore

logger = logging.getLogger("doc_retriever")


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
