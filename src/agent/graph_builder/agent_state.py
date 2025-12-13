from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """State that flows through the agent graph."""
    
    user_id: str                        # User ID for tracking (required)
    query: str                          # User's question
    messages: Annotated[list, add_messages] # Messages
    route: str                          # Which agent(s) to use: "tax", "paye", or "both"
    tax_answer: str                     # Answer from tax policy agent
    paye_answer: str                    # Answer from PAYE agent
    final_answer: str                   # Combined final answer
    sources: list                       # Source documents
    model_used: str                     # Which LLM was used