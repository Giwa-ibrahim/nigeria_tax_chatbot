from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import html
import uuid

def validate_thread_id(v: Any) -> str:
    """Helper to validate if thread_id is a valid UUID4, otherwise generate a new one."""
    if not isinstance(v, str) or not v.strip():
        return str(uuid.uuid4())
    try:
        val = uuid.UUID(v, version=4)
        if str(val) == v:
            return v
    except ValueError:
        pass
    return str(uuid.uuid4())

# Chat Endpoint
class ChatRequest(BaseModel):
    """Request schema for chat endpoint."""
    user_id: str = Field(..., description="User ID for tracking user sessions. MUST be a valid UUID4.")
    query: str = Field( ..., min_length=1, max_length=500, description="User's tax-related question")
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Conversation thread ID for maintaining context")

    @field_validator('user_id')
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        try:
            val = uuid.UUID(v, version=4)
            if str(val) == v:
                return v
        except ValueError:
            pass
        raise ValueError("user_id must be a valid UUID4 string")

    @field_validator('thread_id', mode='before')
    @classmethod
    def apply_thread_id_validation(cls, v: Any) -> str:
        return validate_thread_id(v)
    @field_validator('query')
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return html.escape(v)

class ChatResponse(BaseModel):
    """Response schema for chat endpoint."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    thread_id: str = Field(..., description="Conversation thread ID for maintaining context")
    bot_response: str = Field(..., description="AI-generated answer to the user's query")
    data_source: str = Field(..., description="Route used to generate the answer")
    timestamp: Optional[datetime] = Field(..., description="Timestamp of the response")
    processing_time_sec: Optional[float] = Field(default=None, description="Time taken to generate the response in seconds")

# Conversation History

class ConversationHistoryResponse(BaseModel):
    """Response schema for conversation history."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    thread_id: str = Field(..., description="Conversation thread ID")
    messages: List[Dict[str, Any]] = Field(..., description="List of messages in the conversation")
    message_count: int = Field(..., description="Total number of messages")
    timestamp: Optional[datetime] = Field(..., description="Timestamp of the response")


# List all sessions for a user

class ListSessionResponse(BaseModel):
    """Response schema for listing sessions."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    threads: List[str] = Field(..., description="All Conversation thread IDs")
    

# Delete Endpoint

class DeleteSessionResponse(BaseModel):
    """Response schema for deleting a session."""
    user_id: str = Field(..., description="User ID for tracking user sessions")
    thread_id: str = Field(..., description="Conversation thread ID")
    status: str = Field(..., description="Status of the session")