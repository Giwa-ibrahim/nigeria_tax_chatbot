import logging
from fastapi import APIRouter, HTTPException, status
from typing import Dict, Any, List
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


# @router.put("/edit-query/{user_id}/{thread_id}", response_model=EditResponse)
# async def edit_message(request: EditRequest):
#     """
#     Edit a user message in the conversation history.
    
#     Note: This updates the last user message in the thread with the new query.
#     """
#     try:
#         logger.info(f"✏️ Edit request - User: {request.user_id}, Thread: {request.thread_id}")
        
#         # Get the compiled agent and checkpointer
#         agent = await get_compiled_agent()
#         checkpointer = agent.checkpointer
        
#         if not checkpointer:
#             raise HTTPException(
#                 status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
#                 detail="Conversation memory is not available"
#             )
        
#         # Get current checkpoint
#         config = {"configurable": {"thread_id": request.thread_id}}
#         checkpoint_tuple = await checkpointer.aget_tuple(config)
        
#         if not checkpoint_tuple or not checkpoint_tuple.checkpoint:
#             raise HTTPException(
#                 status_code=status.HTTP_404_NOT_FOUND,
#                 detail=f"No conversation found for thread_id: {request.thread_id}"
#             )
        
#         # Update the checkpoint with the new query
#         logger.info(f"✅ Edit acknowledged for thread {request.thread_id}")
        
#         return EditResponse(
#             user_id=request.user_id,
#             thread_id=request.thread_id,
#             query=request.query,
#             timestamp=datetime.utcnow()
#         )
        
#     except HTTPException:
#         raise
#     except Exception as e:
#         logger.error(f"❌ Error editing message: {str(e)}", exc_info=True)
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=f"An error occurred while editing the message: {str(e)}"
#         )


@router.get("/list-sessions/{user_id}", response_model=ListSessionResponse)
async def list_sessions(user_id: str):
    """
    List all conversation sessions (thread IDs) for a specific user.
    
    This queries the checkpointer database to find all threads associated with the user.
    """
    try:
        logger.info(f"📋 Listing sessions for user: {user_id}")
        
        # Get the compiled agent and checkpointer
        agent = await get_compiled_agent()
        checkpointer = agent.checkpointer
        
        if not checkpointer:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation memory is not available"
            )
        
        # Query the database for all thread_ids
        # The checkpointer stores data in a 'checkpoints' table
        threads = await list_user_threads(checkpointer, user_id)
        
        logger.info(f"✅ Found {len(threads)} sessions for user {user_id}")
        
        return ListSessionResponse(
            user_id=user_id,
            threads=threads
        )
        
    except HTTPException:
        raise
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
    
    This removes all checkpoint data associated with the thread_id.
    """
    try:
        logger.info(f"🗑️ Delete request - User: {user_id}, Thread: {thread_id}")
        
        # Get the compiled agent and checkpointer
        agent = await get_compiled_agent()
        checkpointer = agent.checkpointer
        
        if not checkpointer:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Conversation memory is not available"
            )
        
        # Delete the thread from the checkpointer
        success = await delete_thread(checkpointer, thread_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thread {thread_id} not found or already deleted"
            )
        
        logger.info(f"✅ Thread {thread_id} deleted successfully")
        
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


async def list_user_threads(checkpointer, user_id: str) -> List[str]:
    """
    List all thread IDs from the checkpointer database.
    
    Args:
        checkpointer: LangGraph checkpointer instance
        user_id: User ID (currently not used for filtering, but kept for future use)
        
    Returns:
        List of thread IDs
    """
    try:
        # Access the connection pool from the checkpointer
        conn_pool = checkpointer.conn
        
        # Query the checkpoints table for distinct thread_ids
        async with conn_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    SELECT DISTINCT thread_id 
                    FROM checkpoints 
                    ORDER BY thread_id
                """)
                rows = await cur.fetchall()
                
                # Extract thread_ids from the results
                thread_ids = [row['thread_id'] for row in rows if row.get('thread_id')]
                
                logger.info(f"Found {len(thread_ids)} threads in database")
                return thread_ids
                
    except Exception as e:
        logger.error(f"❌ Error listing threads: {str(e)}", exc_info=True)
        return []


async def delete_thread(checkpointer, thread_id: str) -> bool:
    """
    Delete all checkpoint data for a specific thread.
    
    Args:
        checkpointer: LangGraph checkpointer instance
        thread_id: Thread ID to delete
        
    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        # Access the connection pool from the checkpointer
        conn_pool = checkpointer.conn
        
        async with conn_pool.connection() as conn:
            async with conn.cursor() as cur:
                # Delete from checkpoints table
                await cur.execute("""
                    DELETE FROM checkpoints 
                    WHERE thread_id = %s
                """, (thread_id,))
                
                # Check if any rows were deleted
                deleted_count = cur.rowcount
                
                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} checkpoint(s) for thread {thread_id}")
                    return True
                else:
                    logger.warning(f"No checkpoints found for thread {thread_id}")
                    return False
                    
    except Exception as e:
        logger.error(f"❌ Error deleting thread: {str(e)}", exc_info=True)
        return False
