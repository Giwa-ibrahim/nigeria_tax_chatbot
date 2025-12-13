import logging
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any
from datetime import datetime

from src.api.utilis.schema import (
    ChatRequest,
    ChatResponse,
    ConversationHistoryRequest,
    ConversationHistoryResponse,
    EditRequest,
    EditResponse,
    ListSessionsRequest,
    ListSessionResponse,
    DeleteSessionRequest,
    DeleteSessionResponse
)
from src.agent.main_agent import main_agent
from src.configurations.config import settings

from src.agent.graph_builder.compiled_agent import get_compiled_agent
logger = logging.getLogger("chat_api")

# Create router
router = APIRouter(prefix="/api/v1", tags=["chatbot"])

# ============================================================================
# MAIN CHAT ENDPOINT
# ============================================================================

@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Handle chat requests with the Nigerian Tax Assistant.
    
    **Args:**
    - user_id: User identifier for tracking
    - query: Tax-related question
    - thread_id: Conversation thread for context
    
    **Returns:**
    - ChatResponse with bot answer and metadata
    """
    try:
        logger.info(f"📨 Chat request - User: {request.user_id}, Thread: {request.thread_id}")
        
        # Call the main agent (async)
        result = await main_agent(
            user_id=request.user_id,
            query=request.query,
            return_sources=False,  # Can be made configurable
            thread_id=request.thread_id
        )
        
        # Map the response to ChatResponse schema
        return ChatResponse(
            user_id=request.user_id,
            thread_id=request.thread_id,
            bot_response=result["answer"],
            data_source=result["route_used"],
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"❌ Error in chat endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your request: {str(e)}"
        )


@router.get("/conversation-history/{user_id}/{thread_id}", response_model=ConversationHistoryResponse)
async def get_conversation_history(user_id: str, thread_id: str):
    """Get conversation history for a user/thread."""
    
    agent= await get_compiled_agent()
    checkpointer = agent.checkpointer
    messages= await query_conversation_history(checkpointer, user_id, thread_id)
    message_count= len(messages)
    if not messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation history not found"
        )
    return ConversationHistoryResponse(
        user_id=user_id,
        thread_id=thread_id,
        messages=messages,
        message_count=message_count,
        timestamp=datetime.utcnow()
    )


@router.put("/edit-query/{user_id}/{thread_id}", response_model=EditResponse)
async def edit_message(user_id: str, thread_id: str, request: EditRequest):
    """Edit a user message."""
    pass

@router.get("/list-sessions/{user_id}", response_model=ListSessionResponse)
async def list_sessions(user_id: str):
    """List all sessions for a user."""
    pass

@router.delete("/delete-session/{user_id}/{thread_id}", response_model=DeleteSessionResponse)
async def delete_session(user_id: str, thread_id: str):
    """Delete a conversation session."""
    pass



# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def query_conversation_history(checkpointer, user_id: str, thread_id: str):
    """
    Query conversation history for a thread from the checkpointer.
    
    Args:
        checkpointer: LangGraph checkpointer instance
        user_id: User ID
        thread_id: Thread/session ID
        
    Returns:
        List of message dictionaries with role, content, and timestamp
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        messages = []
        
        if checkpoint_tuple and checkpoint_tuple.checkpoint:
            channel_values = checkpoint_tuple.checkpoint.get('channel_values', {})
            raw_messages = channel_values.get('messages', [])
            logger.info(f"📜 Found {len(raw_messages)} messages in thread {thread_id}")
            
            for msg in raw_messages:
                # Determine role based on message type
                role = "user" if msg.type == "human" else "assistant"
                
                # Append as dictionary, not ConversationHistoryResponse
                messages.append({
                    "role": role,
                    "content": msg.content,
                    "timestamp": checkpoint_tuple.checkpoint.get('ts', datetime.utcnow().isoformat())
                })
        
        return messages
        
    except Exception as e:
        logger.error(f"❌ Error querying conversation history: {str(e)}", exc_info=True)
        return []
