from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# Chat Endpoint
class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    query: str = Field( ..., min_length=1, max_length=1000, description="User's tax-related question")
    thread_id: Optional[str] = Field( default="default", description="Conversation thread ID for maintaining context")

class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    thread_id: str = Field(..., description="Conversation thread ID for maintaining context")
    bot_response: str = Field(..., description="AI-generated answer to the user's query")
    data_source: str = Field(..., description="Route used to generate the answer")
    timestamp: Optional[datetime] = Field(..., description="Timestamp of the response")

# Conversation History

class ConversationHistoryRequest(BaseModel):
    """Request schema for conversation history."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    thread_id: str = Field(..., description="Conversation thread ID for maintaining context")

class ConversationHistoryResponse(BaseModel):
    """Response schema for conversation history."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    thread_id: str = Field(..., description="Conversation thread ID")
    messages: List[Dict[str, Any]] = Field(..., description="List of messages in the conversation")
    message_count: int = Field(..., description="Total number of messages")
    timestamp: Optional[datetime] = Field(..., description="Timestamp of the response")


# Edit/Update User Query
class EditRequest(BaseModel):
    """Request schema for editing user query."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    thread_id: str = Field(..., description="Conversation thread ID for maintaining context")
    query: str = Field(..., description="User's tax-related question")


class EditResponse(BaseModel):
    """Response schema for editing user query."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    thread_id: str = Field(..., description="Conversation thread ID for maintaining context")
    query: str = Field(..., description="User's tax-related question")
    timestamp: Optional[datetime] = Field(..., description="Timestamp of the response")

# List all sessions for a user
class ListSessionsRequest(BaseModel):
    """Request schema for listing sessions."""
    user_id: str = Field(..., description="User ID for tracking user sessions")

class ListSessionResponse(BaseModel):
    """Response schema for listing sessions."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    threads: List[str] = Field(..., description="All Conversation thread IDs")
    

# Delete Endpoint
class DeleteSessionRequest(BaseModel):
    """Request schema for deleting a session."""
    user_id: str = Field(
        ...,
        description="User ID for tracking user sessions"
    )
    thread_id: str = Field(
        ...,
        description="Conversation thread ID for maintaining context"
    )

class DeleteSessionResponse(BaseModel):
    """Response schema for deleting a session."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    thread_id: str = Field(..., description="Conversation thread ID")
    status: str = Field(..., description="Status of the session")