"""
Context Injector — Dynamic user context injection for agent prompts.
"""
from src.agent.graph_builder.agent_state import AgentState


def build_user_context_block(state: AgentState) -> str:
    """
    Return the user profile context block for injection into agent prompts.

    Decision is driven by the router LLM's `needs_user_context` flag — not
    keyword matching — so it handles nuanced cases correctly:
      - "Can my employer deduct from my gross?" → False (policy question)
      - "What is my effective tax rate?" → True (needs actual user data)

    Returns:
        Formatted context string if injection is warranted, empty string otherwise.
    """
    meta = state.get("meta_analysis") or {}
    needs_context = meta.get("needs_user_context", False)

    if not needs_context:
        return ""

    global_ctx = state.get("global_user_context")
    if not global_ctx:
        return ""

    return f"\n{global_ctx}\n"
