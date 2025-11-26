import sys
import os
import chromadb
from pathlib import Path
import logging
from dotenv import load_dotenv
from langchain_chroma import Chroma
from embeddings import get_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
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
            embedding_function=get_embeddings(),
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

def load_documents_to_vectorstore(
    collection_name: str,
    folder_path: str,
    document_type: str,
    force_reindex: bool = False
):
    """
    Generic function to load documents from a folder into a ChromaDB collection.
    
    Args:
        collection_name: Name of the ChromaDB collection
        folder_path: Path to folder containing .txt files
        document_type: Type of documents (e.g., 'tax_policy', 'paye_calculation')
        force_reindex: If True, delete and recreate the collection
    
    Returns:
        The vectorstore instance or None if failed
    """
    folder = Path(folder_path)
    
    # Create vectorstore
    vectorstore = create_vectorstore(collection_name)
    
    if not vectorstore:
        return None
    
    # Check if folder exists
    if not folder.exists():
        logging.warning(f"Folder not found: {folder_path}")
        logging.info(f"Vectorstore '{collection_name}' created (empty)")
        return vectorstore
    
    # Handle force reindex
    if force_reindex:
        logging.info(f"Force reindexing '{collection_name}' - clearing existing documents")
        client = chromadb.PersistentClient(path="./chroma_db")
        try:
            client.delete_collection(collection_name)
            logging.info(f"Deleted existing collection: {collection_name}")
        except:
            pass
        vectorstore = create_vectorstore(collection_name)
        existing_sources = set()
    else:
        existing_sources = get_existing_sources(vectorstore)
        logging.info(f"Found {len(existing_sources)} already indexed documents in '{collection_name}'")
    
    # Process each text file
    files_processed = 0
    for file_path in folder.glob("*.txt"):
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
                    "type": document_type,
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                }
                for i in range(len(chunks))
            ]
            
            # Add to vectorstore
            vectorstore.add_texts(texts=chunks, metadatas=metadatas)
            logging.info(f"Added {file_path.name} ({len(chunks)} chunks)")
            files_processed += 1
            
        except Exception as e:
            logging.error(f"Error loading {file_path.name}: {str(e)}")
    
    if files_processed == 0:
        logging.info(f"No new documents to index in '{collection_name}'")
    else:
        logging.info(f"Finished loading {files_processed} new documents into '{collection_name}'")
    
    return vectorstore


def load_policy_documents(processed_folder: str = "dataset/processed_data/tax_policy", force_reindex: bool = False):
    """Load tax policy documents into ChromaDB."""
    return load_documents_to_vectorstore(
        collection_name="tax_documents",
        folder_path=processed_folder,
        document_type="tax_policy",
        force_reindex=force_reindex
    )


def load_paye_documents(paye_folder: str = "dataset/processed_data/paye_calc", force_reindex: bool = False):
    """Load PAYE calculation documents into ChromaDB."""
    return load_documents_to_vectorstore(
        collection_name="paye_calculations",
        folder_path=paye_folder,
        document_type="paye_calculation",
        force_reindex=force_reindex
    )


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
    
    print("\n" + "="*60)
    print("Loading Tax Policy Documents")
    print("="*60)
    
    # Load tax policy documents
    load_policy_documents(force_reindex=force_reindex)
    
    print("\n" + "="*60)
    print("Loading PAYE Calculation Documents")
    print("="*60)
    
    # Load PAYE documents
    load_paye_documents(force_reindex=force_reindex)
    
    # Test queries
    print("\n" + "="*60)
    print("Testing Tax Policy Query")
    print("="*60)
    results = query_vectorstore("tax_documents", "What are the tax exemptions for individuals?")
    if results:
        print(f"Found {len(results)} relevant chunks\n")
        for i, (doc, score) in enumerate(results):
            print(f"Result {i+1} (score: {score:.4f}):")
            print(f"Source: {doc.metadata['source']}")
            print(f"Text: {doc.page_content[:200]}...\n")
    
    print("\n" + "="*60)
    print("Testing PAYE Query")
    print("="*60)
    results = query_vectorstore("paye_calculations", "How is PAYE calculated?")
    if results:
        print(f"Found {len(results)} relevant chunks\n")
        for i, (doc, score) in enumerate(results):
            print(f"Result {i+1} (score: {score:.4f}):")
            print(f"Source: {doc.metadata['source']}")
            print(f"Text: {doc.page_content[:200]}...\n")