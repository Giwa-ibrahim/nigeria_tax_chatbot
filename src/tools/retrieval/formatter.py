import logging
from typing import List, Dict, Tuple, Optional
from src.agent.prompt_library.rag_prompts import RAG_PROMPT_TEMPLATE
from src.agent.prompt_library.base import get_preference_instructions

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


def create_prompt(query: str, context: str, chat_history: str, user_preferences: Optional[Dict] = None) -> str:
    """
    Create a prompt for the LLM with query and context.
    
    Args:
        query: User query
        context: Retrieved context
        chat_history: Formatted chat history
        user_preferences: Dict containing learned preferences
    
    Returns:
        Formatted prompt
    """
    # Add chat history section if available
    history_section = ""
    if chat_history and chat_history.strip() and chat_history != "No previous conversation.":
        history_section = f"\nPREVIOUS CONVERSATION:\n{chat_history}\n"
    
    preference_instructions = get_preference_instructions(user_preferences)
    
    prompt = RAG_PROMPT_TEMPLATE.format(
        history_section=history_section,
        context=context,
        query=query,
        preference_instructions=preference_instructions
    )
    
    return prompt
