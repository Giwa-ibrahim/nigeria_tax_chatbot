import logging

logger = logging.getLogger("agent_utils")


def format_chat_history(messages: list) -> str:
    """Format chat history for inclusion in prompts."""
    if not messages:
        return "No previous conversation."
    
    formatted = []
    for msg in messages[-10:]:  # Only last 5 exchanges (10 messages)
        role = "user" if msg.type == "human" else "assistant"
        content = msg.content
        
        if role == "user":
            formatted.append(f"User: {content}")
        else:
            formatted.append(f"Assistant: {content}")
    
    return "\n".join(formatted)
