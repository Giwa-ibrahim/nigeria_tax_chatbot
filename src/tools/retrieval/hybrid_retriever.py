import logging
from typing import List, Dict, Tuple, Optional
from rank_bm25 import BM25Okapi
import structlog
from src.vector_db.vectors import create_vectorstore

logger = logging.getLogger("hybrid_retriever")

class HybridRetrieverCache:
    _instances = {}

    @classmethod
    def get_instance(cls, collection_name: str) -> Optional[Tuple[BM25Okapi, List[str], List[Dict]]]:
        if collection_name not in cls._instances:
            logger.info(f"Initializing BM25 index for {collection_name}...")
            try:
                vectorstore = create_vectorstore(collection_name)
                if not vectorstore:
                    return None
                
                # Fetch all documents
                all_docs = vectorstore.get()
                documents = all_docs.get('documents', [])
                metadatas = all_docs.get('metadatas', [])
                
                if not documents:
                    logger.warning(f"No documents found in {collection_name} for BM25 initialization")
                    return None
                
                # Tokenize documents for BM25
                tokenized_corpus = [doc.lower().split() for doc in documents]
                bm25 = BM25Okapi(tokenized_corpus)
                
                cls._instances[collection_name] = (bm25, documents, metadatas)
                logger.info(f"Successfully initialized BM25 index for {collection_name} with {len(documents)} docs")
            except Exception as e:
                logger.error(f"Failed to initialize BM25 for {collection_name}: {str(e)}")
                return None
                
        return cls._instances[collection_name]


def bm25_search(
    query: str, 
    collection_name: str, 
    top_k: int = 10
) -> List[Tuple[str, Dict, float]]:
    """
    Search using BM25.
    Returns: List of tuples (text, metadata, score)
    """
    instance_data = HybridRetrieverCache.get_instance(collection_name)
    if not instance_data:
        return []
        
    bm25, documents, metadatas = instance_data
    
    # Tokenize query
    tokenized_query = query.lower().split()
    
    # Get scores
    doc_scores = bm25.get_scores(tokenized_query)
    
    # Sort and get top k
    top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_k]
    
    results = []
    for idx in top_indices:
        score = doc_scores[idx]
        if score > 0:  # Only include results with some keyword overlap
            results.append((documents[idx], metadatas[idx], score))
            
    return results


def reciprocal_rank_fusion(
    semantic_results: List[Tuple[str, Dict, float]],
    bm25_results: List[Tuple[str, Dict, float]],
    k: int = 60,
    top_k: int = 10
) -> List[Tuple[str, Dict, float]]:
    """
    Combine semantic and BM25 results using Reciprocal Rank Fusion.
    k: Constant for RRF (default 60 is standard in literature)
    """
    # Create a dictionary to hold combined scores by document text
    # Using text as key to identify identical documents
    # Alternatively, we could use an ID if we had one.
    rrf_scores = {}
    doc_mapping = {}
    
    # Process semantic results
    for rank, (doc, meta, _) in enumerate(semantic_results, 1):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0.0
            doc_mapping[doc] = meta
        rrf_scores[doc] += 1.0 / (k + rank)
        
    # Process BM25 results
    for rank, (doc, meta, _) in enumerate(bm25_results, 1):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0.0
            doc_mapping[doc] = meta
        rrf_scores[doc] += 1.0 / (k + rank)
        
    # Sort by RRF score
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return formatted top_k
    fused_results = []
    for doc, rrf_score in sorted_docs[:top_k]:
        fused_results.append((doc, doc_mapping[doc], rrf_score))
        
    return fused_results
