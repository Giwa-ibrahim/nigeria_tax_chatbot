"""
High-level utilities for managing chat sessions.
Provides simple APIs for common operations.
"""
import logging
from datetime import datetime
from typing import List, Dict, Optional, Any
from uuid import uuid4

from src.database.repository import (
    ChatSessionRepository,
    ChatMessageRepository,
    ChatUserRepository
)
from src.database.models import ChatMessage

logger = logging.getLogger("db_utils")


class ChatManager:
    """
    High-level chat management utilities.
    Simplifies common chat operations.
    """
    
    @staticmethod
    async def start_session(
        user_id: Optional[str] = None
    ) -> str:
        """
        Start a new chat session.

        Args:
            user_id: Optional user identifier

        Returns:
            thread_id: Session identifier
        """
        thread_id = str(uuid4())

        await ChatSessionRepository.create_session(
            thread_id=thread_id,
            user_id=user_id
        )

        logger.info(f"Started new session: {thread_id}")
        return thread_id

    
    @staticmethod
    async def add_user_message(
        thread_id: str,
        content: str
    ) -> ChatMessage:
        """
        Add a user message to session.
        
        Args:
            thread_id: Session ID
            content: Message content
        
        Returns:
            Message object
        """
        return await ChatMessageRepository.create_message(
            session_id=thread_id,
            role="user",
            content=content
        )
    
    @staticmethod
    async def add_assistant_message(
        thread_id: str,
        content: str,
        agent_type: Optional[str] = None,
        tokens_used: Optional[int] = None
    ) -> ChatMessage:
        """
        Add an assistant message to session.
        
        Args:
            thread_id: Session ID
            content: Message content
            agent_type: Which agent generated the response
            tokens_used: Number of tokens used
        
        Returns:
            Message object
        """
        return await ChatMessageRepository.create_message(
            session_id=thread_id,
            role="assistant",
            content=content,
            agent_type=agent_type,
            tokens_used=tokens_used
        )
    
    @staticmethod
    async def get_session_history(
        thread_id: str,
        limit: Optional[int] = None,
        format: str = "messages"
    ) -> List[Dict[str, Any]]:
        """
        Get session history.
        
        Args:
            thread_id: Session ID
            limit: Maximum number of messages
            format: Output format ("messages" or "langchain")
        
        Returns:
            List of messages
        """
        messages = await ChatMessageRepository.get_session_messages(
            session_id=thread_id,
            limit=limit
        )
        
        if format == "langchain":
            # Convert to LangChain format
            return [
                {
                    "role": msg.role,
                    "content": msg.content
                }
                for msg in messages
            ]
        else:
            # Return as dictionaries
            return [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "agent_type": msg.agent_type,
                    "tokens_used": msg.tokens_used,
                    "created_at": msg.created_at.isoformat()
                }
                for msg in messages
            ]
    
    @staticmethod
    async def get_recent_context(
        thread_id: str,
        message_count: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get recent messages for context window.
        Returns in LangChain format for agent consumption.
        
        Args:
            thread_id: Session ID
            message_count: Number of recent messages
        
        Returns:
            List of messages in LangChain format
        """
        messages = await ChatMessageRepository.get_recent_messages(
            session_id=thread_id,
            count=message_count
        )
        
        return [
            {
                "role": msg.role,
                "content": msg.content
            }
            for msg in messages
        ]
    
    @staticmethod
    async def end_session(thread_id: str) -> bool:
        """
        End/archive a session.
        
        Args:
            thread_id: Session ID
        
        Returns:
            Success status
        """
        success = await ChatSessionRepository.update_session_status(
            thread_id=thread_id,
            status="archived"
        )
        
        if success:
            logger.info(f"Session {thread_id} archived")
        
        return success
    
    @staticmethod
    async def get_session_stats(thread_id: str) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Args:
            thread_id: Session ID
        
        Returns:
            Statistics dictionary
        """
        chat_session = await ChatSessionRepository.get_session(thread_id)
        
        if not chat_session:
            return {}
        
        message_stats = await ChatMessageRepository.get_message_stats(thread_id)
        
        return {
            "thread_id": thread_id,
            "status": chat_session.status,
            "created_at": chat_session.created_at.isoformat(),
            "updated_at": chat_session.updated_at.isoformat(),
            "message_count": chat_session.message_count,
            "total_tokens": message_stats.get("total_tokens", 0),
            "agent_distribution": message_stats.get("agent_distribution", {}),
            "user_id": chat_session.user_id
        }
    
    @staticmethod
    async def check_user_rate_limit(
        user_id: str,
        max_requests: int = 60,
        window_minutes: int = 1
    ) -> tuple[bool, int]:
        """
        Check if user is within rate limits.
        
        Args:
            user_id: User identifier
            max_requests: Maximum requests allowed
            window_minutes: Time window in minutes
        
        Returns:
            (is_allowed, remaining_requests)
        """
        return await ChatUserRepository.check_rate_limit(
            user_id=user_id,
            max_requests=max_requests,
            window_minutes=window_minutes
        )
    
    @staticmethod
    async def track_user_activity(user_id: str):
        """
        Track user activity.
        
        Args:
            user_id: User identifier
        """
        await ChatUserRepository.create_or_update_user(user_id=user_id)
