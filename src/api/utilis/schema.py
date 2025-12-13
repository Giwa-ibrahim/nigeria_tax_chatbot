from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# Chat Endpoint
class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    user_id: str = Field(..., description="User ID for tracking user sessions", example="user_12345")
    query: str = Field( ..., min_length=1, max_length=1000, description="User's tax-related question", example="What is the current VAT rate in Nigeria?")
    thread_id: Optional[str] = Field( default="default", description="Conversation thread ID for maintaining context", example="user_12345")

class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    user_id: str = Field(..., description="User ID for tracking user sessions", example="user_12345")
    thread_id: str = Field(..., description="Conversation thread ID for maintaining context", example="user_12345")
    bot_response: str = Field(..., description="AI-generated answer to the user's query", example="The current VAT rate in Nigeria is 5%...")
    data_source: str = Field(..., description="Route used to generate the answer", example="tax_policy_agent")
    timestamp: Optional[datetime] = Field(..., description="Timestamp of the response", example="2023-04-01T12:34:56.789Z")

# Conversation History

class ConversationHistoryRequest(BaseModel):
    """Request schema for conversation history."""
    user_id: str = Field(..., description="User ID for tracking user sessions", example="user_12345")
    thread_id: str = Field(..., description="Conversation thread ID for maintaining context", example="user_12345")

class ConversationHistoryResponse(BaseModel):
    """Response schema for conversation history."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    thread_id: str = Field(..., description="Conversation thread ID")
    messages: List[Dict[str, Any]] = Field(..., description="List of messages in the conversation")
    message_count: int = Field(..., description="Total number of messages")
    timestamp: Optional[datetime] = Field(..., description="Timestamp of the response", example="2023-04-01T12:34:56.789Z")


# Edit/Update User Query
class EditRequest(BaseModel):
    """Request schema for editing user query."""
    user_id: str = Field(..., description="User ID for tracking user sessions", example="user_12345")
    thread_id: str = Field(..., description="Conversation thread ID for maintaining context", example="user_12345")
    query: str = Field(..., description="User's tax-related question", example="What is the current VAT rate in Nigeria?")


class EditResponse(BaseModel):
    """Response schema for editing user query."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    thread_id: str = Field(..., description="Conversation thread ID for maintaining context")
    query: str = Field(..., description="User's tax-related question")
    timestamp: Optional[datetime] = Field(..., description="Timestamp of the response", example="2023-04-01T12:34:56.789Z")

# List all sessions for a user
class ListSessionsRequest(BaseModel):
    """Request schema for listing sessions."""
    user_id: str = Field(..., description="User ID for tracking user sessions", example="user_12345")

class ListSessionResponse(BaseModel):
    """Response schema for listing sessions."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    threads: List[str] = Field(..., description="All Conversation thread IDs")
    

# Delete Endpoint
class DeleteSessionRequest(BaseModel):
    """Request schema for deleting a session."""
    user_id: str = Field(
        ...,
        description="User ID for tracking user sessions",
        example="user_12345"
    )
    thread_id: str = Field(
        ...,
        description="Conversation thread ID for maintaining context",
        example="user_12345"
    )

class DeleteSessionResponse(BaseModel):
    """Response schema for deleting a session."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    thread_id: str = Field(..., description="Conversation thread ID")
    status: str = Field(..., description="Status of the session")