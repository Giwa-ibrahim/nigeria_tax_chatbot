import logging
from typing import List, Dict, Tuple

logger = logging.getLogger("retrieval_formatter")


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
    
    
    prompt = f"""You are a helpful Nigerian Tax Assistant. Use the following context from official tax documents to answer the user's question accurately and concisely.

{history_section}

CONTEXT:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Be CONCISE and DIRECT - focus on answering the specific question asked
3. Prioritize the most relevant information - avoid unnecessary details
4. Use bullet points or short paragraphs for clarity
5. Cite specific tax rates, laws, or regulations when directly relevant
6. If calculations are involved, show only the essential steps
7. DO NOT reference documents by number (e.g., "Document 1", "Document 2")
8. DO NOT repeat the same information multiple times
9. Keep your response focused and to-the-point
10. If the context doesn't contain enough information, say so briefly

ANSWER:"""
    
    return prompt
