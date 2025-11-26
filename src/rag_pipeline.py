import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

# Import our custom modules
from llm import LLMManager
from vectors import (
    load_policy_documents,
    load_paye_documents,
    query_vectorstore,
    create_vectorstore
)
from embeddings import get_embeddings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)


class RAGPipeline:
    """
    RAG (Retrieval-Augmented Generation) Pipeline for Nigerian Tax Chatbot.
    
    This pipeline:
    1. Retrieves relevant documents from vector stores (tax policy & PAYE)
    2. Augments the user query with retrieved context
    3. Generates responses using LLM with the augmented context
    """
    
    def __init__(
        self,
        tax_collection: str = "tax_documents",
        paye_collection: str = "paye_calculations",
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_k: int = 3
    ):
        """
        Initialize the RAG Pipeline.
        
        Args:
            tax_collection: Name of the tax policy collection
            paye_collection: Name of the PAYE collection
            temperature: LLM temperature for response generation
            max_tokens: Maximum tokens in LLM response
            top_k: Number of relevant documents to retrieve
        """
        self.tax_collection = tax_collection
        self.paye_collection = paye_collection
        self.top_k = top_k
        
        # Initialize LLM Manager
        logging.info("Initializing LLM Manager...")
        self.llm_manager = LLMManager(
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Initialize vector stores
        logging.info("Initializing vector stores...")
        self.tax_vectorstore = create_vectorstore(tax_collection)
        self.paye_vectorstore = create_vectorstore(paye_collection)
        
        logging.info("✅ RAG Pipeline initialized successfully")
    
    def retrieve_context(
        self,
        query: str,
        collection_type: str = "both",
        top_k: Optional[int] = None
    ) -> List[Tuple[str, Dict, float]]:
        """
        Retrieve relevant documents from vector stores.
        
        Args:
            query: User query
            collection_type: Which collection to search ("tax", "paye", or "both")
            top_k: Number of documents to retrieve (overrides default)
        
        Returns:
            List of tuples: (document_text, metadata, similarity_score)
        """
        k = top_k or self.top_k
        results = []
        
        try:
            # Query tax policy documents
            if collection_type in ["tax", "both"]:
                logging.info(f"Querying tax policy documents (top {k})...")
                tax_results = query_vectorstore(self.tax_collection, query, top_k=k)
                
                if tax_results:
                    for doc, score in tax_results:
                        results.append((doc.page_content, doc.metadata, score))
                    logging.info(f"Retrieved {len(tax_results)} tax policy documents")
            
            # Query PAYE documents
            if collection_type in ["paye", "both"]:
                logging.info(f"Querying PAYE documents (top {k})...")
                paye_results = query_vectorstore(self.paye_collection, query, top_k=k)
                
                if paye_results:
                    for doc, score in paye_results:
                        results.append((doc.page_content, doc.metadata, score))
                    logging.info(f"Retrieved {len(paye_results)} PAYE documents")
            
            # Sort by similarity score (lower is better for distance metrics)
            results.sort(key=lambda x: x[2])
            
            logging.info(f"Total documents retrieved: {len(results)}")
            return results
            
        except Exception as e:
            logging.error(f"Error retrieving context: {str(e)}")
            return []
    
    def format_context(self, retrieved_docs: List[Tuple[str, Dict, float]]) -> str:
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
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the LLM with query and context.
        
        Args:
            query: User query
            context: Retrieved context
        
        Returns:
            Formatted prompt
        """
        prompt = f"""You are a helpful Nigerian Tax Assistant. Use the following context from official tax documents to answer the user's question accurately and comprehensively.

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

ANSWER:"""
        
        return prompt
    
    def query(
        self,
        user_query: str,
        collection_type: str = "both",
        top_k: Optional[int] = None,
        force_fallback: bool = False,
        return_sources: bool = False
    ) -> Dict:
        """
        Main RAG pipeline query method.
        
        Args:
            user_query: User's question
            collection_type: Which collection to search ("tax", "paye", or "both")
            top_k: Number of documents to retrieve
            force_fallback: Force use of Groq instead of Gemini
            return_sources: Include source documents in response
        
        Returns:
            Dictionary containing:
                - answer: Generated response
                - sources: List of source documents (if return_sources=True)
                - model_used: Name of the LLM model used
        """
        logging.info(f"Processing query: '{user_query[:100]}...'")
        
        try:
            # Step 1: Retrieve relevant documents
            retrieved_docs = self.retrieve_context(
                query=user_query,
                collection_type=collection_type,
                top_k=top_k
            )
            
            if not retrieved_docs:
                logging.warning("No relevant documents found")
                return {
                    "answer": "I couldn't find relevant information in the tax documents to answer your question. Please try rephrasing or ask about Nigerian tax policies, PAYE calculations, or tax regulations.",
                    "sources": [],
                    "model_used": None
                }
            
            # Step 2: Format context
            context = self.format_context(retrieved_docs)
            
            # Step 3: Create prompt
            prompt = self.create_prompt(user_query, context)
            
            # Step 4: Generate response using LLM
            logging.info("Generating response with LLM...")
            llm = self.llm_manager.get_llm(force_fallback=force_fallback)
            response = llm.invoke(prompt)
            
            # Extract answer
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # Get model name
            model_used = self.llm_manager.get_active_model()
            
            logging.info(f"✅ Response generated successfully using {model_used}")
            
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
            logging.error(f"Error in RAG pipeline: {str(e)}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "model_used": None
            }
    
    def batch_query(
        self,
        queries: List[str],
        collection_type: str = "both",
        top_k: Optional[int] = None
    ) -> List[Dict]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of user queries
            collection_type: Which collection to search
            top_k: Number of documents to retrieve per query
        
        Returns:
            List of response dictionaries
        """
        logging.info(f"Processing {len(queries)} queries in batch...")
        results = []
        
        for i, query in enumerate(queries, 1):
            logging.info(f"Processing query {i}/{len(queries)}")
            result = self.query(
                user_query=query,
                collection_type=collection_type,
                top_k=top_k
            )
            results.append(result)
        
        logging.info(f"✅ Batch processing complete: {len(results)} queries processed")
        return results


def initialize_pipeline(
    force_reindex: bool = False,
    tax_folder: str = "dataset/processed_data/tax_policy",
    paye_folder: str = "dataset/processed_data/paye_calc"
) -> RAGPipeline:
    """
    Initialize the RAG pipeline and load documents if needed.
    
    Args:
        force_reindex: Force reindexing of all documents
        tax_folder: Path to tax policy documents
        paye_folder: Path to PAYE documents
    
    Returns:
        Initialized RAGPipeline instance
    """
    logging.info("="*60)
    logging.info("Initializing RAG Pipeline")
    logging.info("="*60)
    
    # Load documents into vector stores
    logging.info("\n📚 Loading Tax Policy Documents...")
    load_policy_documents(processed_folder=tax_folder, force_reindex=force_reindex)
    
    logging.info("\n📚 Loading PAYE Documents...")
    load_paye_documents(paye_folder=paye_folder, force_reindex=force_reindex)
    
    # Create and return pipeline
    logging.info("\n🚀 Creating RAG Pipeline...")
    pipeline = RAGPipeline()
    
    logging.info("="*60)
    logging.info("✅ RAG Pipeline Ready!")
    logging.info("="*60)
    
    return pipeline


# Example usage
if __name__ == "__main__":
    import sys
    
    # Check for --force flag
    force_reindex = "--force" in sys.argv
    
    # Initialize pipeline
    pipeline = initialize_pipeline(force_reindex=force_reindex)
    
    # Example queries
    print("\n" + "="*60)
    print("EXAMPLE 1: Tax Policy Query")
    print("="*60)
    
    result1 = pipeline.query(
        user_query="What are the tax exemptions for individuals in Nigeria?",
        collection_type="tax",
        return_sources=True
    )
    
    print(f"\n📝 Question: What are the tax exemptions for individuals in Nigeria?")
    print(f"\n🤖 Answer ({result1['model_used']}):")
    print(result1['answer'])
    
    if result1.get('sources'):
        print(f"\n📚 Sources ({len(result1['sources'])} documents):")
        for i, source in enumerate(result1['sources'], 1):
            print(f"  {i}. {source['source']} (score: {source['score']:.4f})")
    
    print("\n" + "="*60)
    print("EXAMPLE 2: PAYE Query")
    print("="*60)
    
    result2 = pipeline.query(
        user_query="How is PAYE calculated for an employee earning 500,000 Naira monthly?",
        collection_type="paye",
        return_sources=True
    )
    
    print(f"\n📝 Question: How is PAYE calculated for an employee earning 500,000 Naira monthly?")
    print(f"\n🤖 Answer ({result2['model_used']}):")
    print(result2['answer'])
    
    if result2.get('sources'):
        print(f"\n📚 Sources ({len(result2['sources'])} documents):")
        for i, source in enumerate(result2['sources'], 1):
            print(f"  {i}. {source['source']} (score: {source['score']:.4f})")
    
    print("\n" + "="*60)
    print("EXAMPLE 3: Combined Query")
    print("="*60)
    
    result3 = pipeline.query(
        user_query="What are the tax reliefs available and how do they affect PAYE?",
        collection_type="both",
        return_sources=True
    )
    
    print(f"\n📝 Question: What are the tax reliefs available and how do they affect PAYE?")
    print(f"\n🤖 Answer ({result3['model_used']}):")
    print(result3['answer'])
    
    if result3.get('sources'):
        print(f"\n📚 Sources ({len(result3['sources'])} documents):")
        for i, source in enumerate(result3['sources'], 1):
            print(f"  {i}. {source['source']} ({source['type']}) - score: {source['score']:.4f}")
    
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Queries")
    print("="*60)
    
    batch_queries = [
        "What is VAT in Nigeria?",
        "How do I calculate personal income tax?",
        "What are the penalties for late tax payment?"
    ]
    
    batch_results = pipeline.batch_query(batch_queries, collection_type="both")
    
    for i, (query, result) in enumerate(zip(batch_queries, batch_results), 1):
        print(f"\n{i}. {query}")
        print(f"   Model: {result['model_used']}")
        print(f"   Answer: {result['answer'][:150]}...")
