import structlog
from fastapi import APIRouter, HTTPException, status, Request
from typing import Dict, Any, List
from datetime import datetime

from src.api.utilis.schema import (
    ChatRequest,
    ChatResponse,
    ConversationHistoryResponse,
    ListSessionResponse,
    DeleteSessionResponse
)
from src.agent.main_agent import main_agent
from src.configurations.config import settings
from src.database.chat_manager import ChatManager
from src.database.repository import ChatSessionRepository
from src.api.utilis.limiter import limiter

logger = structlog.get_logger("chat_api")

# Create router
router = APIRouter(prefix="/api/v1", tags=["chatbot"])

# ============================================================================
# MAIN CHAT ENDPOINT
# ============================================================================

@router.post("/chat", response_model=ChatResponse, status_code=status.HTTP_200_OK)
@limiter.limit("10/minute")
async def chat(request: Request, chat_req: ChatRequest) -> ChatResponse:
    """
    Handle chat requests with the Nigerian Tax Assistant.
    
    🆕 PERSONALIZATION: Now saves messages to custom chat database!
    - User messages stored in chat_messages table
    - Assistant responses tracked with agent type
    - Session tracking in chat_sessions table
    
    **Args:**
    - user_id: User identifier for tracking
    - query: Tax-related question
    - thread_id: Conversation thread for context
    
    **Returns:**
    - ChatResponse with bot answer and metadata
    """
    try:
        logger.info(f"📨 Chat request - User: {chat_req.user_id}, Thread: {chat_req.thread_id}")
        
        # 🆕 CHECK RATE LIMIT using ChatManager
        is_allowed, _ = await ChatManager.check_user_rate_limit(
            user_id=chat_req.user_id,
            max_requests=20,  # 20 requests per hour
            window_minutes=60
        )
        
        if not is_allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Try again later."
            )
            
        # Optional: track user activity
        await ChatManager.track_user_activity(chat_req.user_id)
        
        # 🆕 SAVE USER MESSAGE TO DATABASE
        try:
            await ChatManager.add_user_message(
                thread_id=chat_req.thread_id,
                content=chat_req.query
            )
            logger.info("💾 User message saved to database")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save user message: {e}")
        
        # Call the main agent (async)
        result = await main_agent(
            user_id=chat_req.user_id,
            query=chat_req.query,
            return_sources=False,  # Can be made configurable
            thread_id=chat_req.thread_id
        )
        
        # 🆕 SAVE ASSISTANT RESPONSE TO DATABASE
        try:
            await ChatManager.add_assistant_message(
                thread_id=chat_req.thread_id,
                content=result["answer"],
                agent_type=result["route_used"],  # paye, tax_policy, financial_advice
                tokens_used=0  # TODO: Add token counting in Phase 7
            )
            logger.info("💾 Assistant response saved to database")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save assistant message: {e}")
        
        # Map the response to ChatResponse schema
        return ChatResponse(
            user_id=chat_req.user_id,
            thread_id=chat_req.thread_id,
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
    try:
        messages = await ChatManager.get_session_history(thread_id, limit=50)
        message_count = len(messages)
        if not messages:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation history not found"
            )
            
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", ""),
                "timestamp": msg.get("created_at", datetime.utcnow().isoformat())
            })

        return ConversationHistoryResponse(
            user_id=user_id,
            thread_id=thread_id,
            messages=formatted_messages,
            message_count=message_count,
            timestamp=datetime.utcnow()
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting conversation history: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while fetching history: {str(e)}"
        )


@router.get("/list-sessions/{user_id}", response_model=ListSessionResponse)
async def list_sessions(user_id: str, limit: int = 10):
    """
    List all conversation sessions (thread IDs) for a specific user.
    """
    try:
        logger.info(f"📋 Listing sessions for user: {user_id}")
        
        
        sessions = await ChatSessionRepository.get_user_sessions(user_id, limit)
        
        # Return list of thread IDs to match schema ListSessionResponse
        threads = [s.id for s in sessions]
        
        logger.info(f"✅ Found {len(threads)} sessions for user {user_id}")
        
        return ListSessionResponse(
            user_id=user_id,
            threads=threads
        )
        
    except Exception as e:
        logger.error(f"❌ Error listing sessions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while listing sessions: {str(e)}"
        )


@router.delete("/delete-session/{user_id}/{thread_id}", response_model=DeleteSessionResponse)
async def delete_session(user_id: str, thread_id: str):
    """
    Delete a conversation session (thread) for a user.
    """
    try:
        logger.info(f"🗑️ Delete request - User: {user_id}, Thread: {thread_id}")
        
        # Delete the session using ChatManager
        success = await ChatManager.end_session(thread_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thread {thread_id} not found or already deleted"
            )
        
        logger.info(f"✅ Thread {thread_id} deleted (archived) successfully")
        
        return DeleteSessionResponse(
            user_id=user_id,
            thread_id=thread_id,
            status="deleted"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting session: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while deleting the session: {str(e)}"
        )



