from typing import TypedDict, Annotated, Optional
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State that flows through the agent graph."""

    user_id: str                        # User ID for tracking (required)
    query: str                          # User's question
    messages: Annotated[list, add_messages] # Messages
    route: str                          # Which agent(s) to use: "tax", "paye", "financial", or "both"
    user_profile: dict                  # Pre-loaded main app profile data
    user_preferences: dict              # Learned user preferences
    tax_answer: str                     # Answer from tax policy agent
    paye_answer: str                    # Answer from PAYE agent
    final_answer: str                   # Combined final answer
    sources: list                       # Source documents
    model_used: str                     # Which LLM was used

    # Unified user context (built once by ContextPreparator, injected dynamically by agents)
    global_user_context: Optional[str]  # Formatted profile/income/PAYE block — None if no data

    # Combined router output (route + meta-analysis + needs_user_context in one LLM call)
    meta_analysis: Optional[dict]       # {route, needs_user_context, is_calculation_request, ...}