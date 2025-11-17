import sys
import chromadb
from pathlib import Path
import logging
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Initialize embeddings
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

def create_vectorstore(collection_name: str, persist_directory: str = "./chroma_db"):
    """Create or load a Chroma vectorstore."""
    try:
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory
        )
        logging.info(f"Vectorstore '{collection_name}' ready")
        return vectorstore
    except Exception as e:
        logging.error(f"Error creating vectorstore: {str(e)}")
        return None

def get_existing_sources(vectorstore):
    """Get list of already indexed document sources."""
    try:
        # Get all documents from the collection
        all_docs = vectorstore.get()
        if all_docs and all_docs['metadatas']:
            sources = set(meta.get('source') for meta in all_docs['metadatas'] if meta.get('source'))
            return sources
        return set()
    except Exception as e:
        logging.error(f"Error getting existing sources: {str(e)}")
        return set()

def load_processed_documents(processed_folder: str, force_reindex: bool = False):
    """Load all text files from processed_data folder into ChromaDB."""
    processed_path = Path(processed_folder)
    
    # Create vectorstore for tax documents
    tax_vectorstore = create_vectorstore("tax_documents")
    
    if not tax_vectorstore:
        return
    
    # Get already indexed documents (unless forcing reindex)
    if force_reindex:
        logging.info("Force reindexing - clearing existing documents")
        # Delete and recreate collection
        client = chromadb.PersistentClient(path="./chroma_db")
        try:
            client.delete_collection("tax_documents")
            logging.info("Deleted existing collection")
        except:
            pass
        tax_vectorstore = create_vectorstore("tax_documents")
        existing_sources = set()
    else:
        existing_sources = get_existing_sources(tax_vectorstore)
        logging.info(f"Found {len(existing_sources)} already indexed documents")
    
    # Process each text file
    files_processed = 0
    for file_path in processed_path.glob("*.txt"):
        # Skip if already indexed
        if file_path.name in existing_sources:
            logging.info(f"Skipped (already indexed): {file_path.name}")
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Split into chunks
            chunks = text_splitter.split_text(text)
            
            # Create metadata for each chunk
            metadatas = [
                {
                    "source": file_path.name,
                    "type": "tax_policy",
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                for i in range(len(chunks))
            ]
            
            # Add to vectorstore
            tax_vectorstore.add_texts(texts=chunks, metadatas=metadatas)
            logging.info(f"Added {file_path.name} ({len(chunks)} chunks)")
            files_processed += 1
            
        except Exception as e:
            logging.error(f"Error loading {file_path.name}: {str(e)}")
    
    if files_processed == 0:
        logging.info("No new documents to index")
    else:
        logging.info(f"Finished loading {files_processed} new documents into ChromaDB")

def create_paye_vectorstore(paye_text: str = None):
    """Create a separate vectorstore for PAYE calculations and rules."""
    paye_vectorstore = create_vectorstore("paye_rules")
    
    if paye_vectorstore and paye_text:
        chunks = text_splitter.split_text(paye_text)
        metadatas = [{
                "type": "paye", 
                "source": "paye_rules"} for _ in chunks]
        paye_vectorstore.add_texts(texts=chunks, metadatas=metadatas)
        logging.info("PAYE vectorstore created and populated")
    else:
        logging.info("PAYE vectorstore created (empty - ready for future data)")
    
    return paye_vectorstore

def query_vectorstore(collection_name: str, query_text: str, top_k: int = 3):
    """Query a vectorstore and return relevant documents."""
    try:
        vectorstore = create_vectorstore(collection_name)
        results = vectorstore.similarity_search_with_score(query_text, k=top_k)
        return results
    except Exception as e:
        logging.error(f"Error querying vectorstore: {str(e)}")
        return None

if __name__ == "__main__":
    # Check for --force flag to reindex everything
    force_reindex = "--force" in sys.argv
    
    # Load processed documents into ChromaDB
    processed_folder = "dataset/processed_data"
    
    # Load existing processed documents
    load_processed_documents(processed_folder=processed_folder, force_reindex=force_reindex)
    
    # Create empty PAYE vectorstore for later use
    create_paye_vectorstore()
    
    # Test query
    print("\n--- Testing Query ---")
    results = query_vectorstore("tax_documents", "What are the tax exemptions for individuals?")
    if results:
        print(f"Found {len(results)} relevant chunks\n")
        for i, (doc, score) in enumerate(results):
            print(f"Result {i+1} (score: {score:.4f}):")
            print(f"Source: {doc.metadata['source']}")
            print(f"Text: {doc.page_content[:200]}...\n")