import asyncio
import logging
from src.tools.retrieval.retriever import retrieve_context
from src.configurations.config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_hybrid_rag")

def run_eval():
    queries = [
        # Keyword-heavy queries where BM25 shines
        "Section 24",
        "PITA 2011",
        "What is the penalty under section 47?",
        "Consolidated relief allowance",
        "Form A",
        
        # Semantic-heavy queries where embeddings shine
        "How do I calculate my tax?",
        "What happens if I don't pay tax on time?",
        "Tell me about the benefits of paying tax",
        "Are there exemptions for freelancers?",
        "Who is responsible for deducting PAYE?"
    ]
    
    print("="*60)
    print("Evaluating PURE SEMANTIC SEARCH")
    print("="*60)
    
    for query in queries:
        results = retrieve_context(query, collection_type="tax", use_hybrid=False, top_k=3)
        print(f"\nQuery: '{query}'")
        for i, (doc, meta, score) in enumerate(results):
            print(f"  {i+1}. [Score: {score:.3f}] {doc[:100]}...")
            
    print("\n" + "="*60)
    print("Evaluating HYBRID SEARCH (Semantic + BM25)")
    print("="*60)
    
    for query in queries:
        results = retrieve_context(query, collection_type="tax", use_hybrid=True, top_k=3)
        print(f"\nQuery: '{query}'")
        for i, (doc, meta, score) in enumerate(results):
            print(f"  {i+1}. [Score: {score:.3f}] {doc[:100]}...")
            
if __name__ == "__main__":
    run_eval()
