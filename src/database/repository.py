"""
Repository layer for database operations.
Simplified for chat_sessions, chat_messages, chat_summaries, chat_users.
"""
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from uuid import uuid4

from sqlalchemy import select, func, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import ChatSession, ChatMessage, ChatSummary, ChatUser
from src.database.connection import get_db_session

logger = logging.getLogger("chat_respositry")


class ChatSessionRepository:
    """Repository for chat session operations."""
    
    @staticmethod
    async def create_session(
        thread_id: str,
        user_id: Optional[str] = None
    ) -> ChatSession:
        """Create a new chat session."""
        async with get_db_session() as session:
            chat_session = ChatSession(
                id=thread_id,
                user_id=user_id,
                status="active",
                message_count=0,
                current_state={"status": "active", "message_count": 0},
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            session.add(chat_session)
            await session.commit()
            logger.info(f"Created session: {thread_id}")
            return chat_session
    
    @staticmethod
    async def get_session(thread_id: str) -> Optional[ChatSession]:
        """Get session by ID."""
        async with get_db_session() as session:
            stmt = select(ChatSession).where(ChatSession.id == thread_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
    
    @staticmethod
    async def get_user_sessions(
        user_id: str,
        limit: int = 10,
        status: str = "active"
    ) -> List[ChatSession]:
        """Get all sessions for a user."""
        async with get_db_session() as session:
            stmt = (
                select(ChatSession)
                .where(
                    and_(
                        ChatSession.user_id == user_id,
                        ChatSession.status == status
                    )
                )
                .order_by(desc(ChatSession.updated_at))
                .limit(limit)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())
    
    @staticmethod
    async def update_session_status(thread_id: str, status: str) -> bool:
        """Update session status."""
        async with get_db_session() as session:
            stmt = select(ChatSession).where(ChatSession.id == thread_id)
            result = await session.execute(stmt)
            chat_session = result.scalar_one_or_none()
            
            if chat_session:
                chat_session.status = status
                chat_session.updated_at = datetime.utcnow()
                await session.commit()
                return True
            return False
    
    @staticmethod
    async def delete_old_sessions(days: int = 30) -> int:
        """Delete sessions older than specified days."""
        async with get_db_session() as session:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            stmt = select(ChatSession).where(
                and_(
                    ChatSession.updated_at < cutoff_date,
                    ChatSession.status != "active"
                )
            )
            result = await session.execute(stmt)
            sessions = result.scalars().all()
            
            count = len(sessions)
            for sess in sessions:
                await session.delete(sess)
            
            await session.commit()
            logger.info(f"Deleted {count} old sessions")
            return count


class ChatMessageRepository:
    """Repository for chat message operations."""
    
    @staticmethod
    async def create_message(
        session_id: str,
        role: str,
        content: str,
        agent_type: Optional[str] = None,
        tokens_used: Optional[int] = None
    ) -> ChatMessage:
        """Create a new message."""
        async with get_db_session() as session:
            message = ChatMessage(
                id=str(uuid4()),
                session_id=session_id,
                role=role,
                content=content,
                agent_type=agent_type,
                tokens_used=tokens_used,
                created_at=datetime.utcnow()
            )
            session.add(message)

            # Update session message count and current_state
            stmt = select(ChatSession).where(ChatSession.id == session_id)
            result = await session.execute(stmt)
            chat_session = result.scalar_one_or_none()

            if chat_session:
                chat_session.message_count += 1
                chat_session.updated_at = datetime.utcnow()
                chat_session.current_state = {
                    "status": "active",
                    "message_count": chat_session.message_count,
                    "last_role": role,
                    "last_agent": agent_type
                }

            await session.commit()
            return message
    
    @staticmethod
    async def get_session_messages(
        session_id: str,
        limit: Optional[int] = None
    ) -> List[ChatMessage]:
        """Get all messages for a session."""
        async with get_db_session() as session:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(ChatMessage.created_at)
            )
            
            if limit:
                stmt = stmt.limit(limit)
            
            result = await session.execute(stmt)
            return list(result.scalars().all())
    
    @staticmethod
    async def get_recent_messages(
        session_id: str,
        count: int = 10
    ) -> List[ChatMessage]:
        """Get N most recent messages for context."""
        async with get_db_session() as session:
            stmt = (
                select(ChatMessage)
                .where(ChatMessage.session_id == session_id)
                .order_by(desc(ChatMessage.created_at))
                .limit(count)
            )
            result = await session.execute(stmt)
            messages = list(result.scalars().all())
            return list(reversed(messages))  # Chronological order
    
    @staticmethod
    async def get_message_stats(session_id: str) -> Dict[str, Any]:
        """Get message statistics for a session."""
        async with get_db_session() as session:
            # Total messages
            total_stmt = select(func.count(ChatMessage.id)).where(
                ChatMessage.session_id == session_id
            )
            total_result = await session.execute(total_stmt)
            total_messages = total_result.scalar()
            
            # Total tokens
            tokens_stmt = select(func.sum(ChatMessage.tokens_used)).where(
                ChatMessage.session_id == session_id
            )
            tokens_result = await session.execute(tokens_stmt)
            total_tokens = tokens_result.scalar() or 0
            
            # Agent distribution
            agent_stmt = (
                select(ChatMessage.agent_type, func.count(ChatMessage.id))
                .where(ChatMessage.session_id == session_id)
                .group_by(ChatMessage.agent_type)
            )
            agent_result = await session.execute(agent_stmt)
            agent_distribution = dict(agent_result.all())
            
            return {
                "total_messages": total_messages,
                "total_tokens": total_tokens,
                "agent_distribution": agent_distribution
            }

class ChatSummaryRepository:
    """Repository for chat summary operations."""

    @staticmethod
    async def create_summary(
        session_id: str,
        summary_text: str,
        message_range: Optional[str] = None
    ) -> ChatSummary:
        """Persist a conversation summary to the chat_summaries table."""
        async with get_db_session() as session:
            summary = ChatSummary(
                id=str(uuid4()),
                session_id=session_id,
                summary_text=summary_text,
                message_range=message_range,
                created_at=datetime.utcnow()
            )
            session.add(summary)
            await session.commit()
            logger.info(f"Saved summary for session: {session_id} (range: {message_range})")
            return summary


class ChatUserRepository:
    """Repository for user tracking operations."""

    @staticmethod
    async def create_or_update_user(
        user_id: str
    ) -> ChatUser:
        """Create or update user record."""
        async with get_db_session() as session:
            stmt = select(ChatUser).where(ChatUser.user_id == user_id)
            result = await session.execute(stmt)
            chat_user = result.scalar_one_or_none()
            
            if chat_user:
                # Update existing
                chat_user.last_activity = datetime.utcnow()
                chat_user.requests_count += 1
                chat_user.last_request_time = datetime.utcnow()
            else:
                # Create new
                chat_user = ChatUser(
                    id=str(uuid4()),
                    user_id=user_id,
                    last_activity=datetime.utcnow(),
                    requests_count=1,
                    last_request_time=datetime.utcnow()
                )
                session.add(chat_user)
            
            await session.commit()
            return chat_user
    
    @staticmethod
    async def check_rate_limit(
        user_id: str,
        max_requests: int = 60,
        window_minutes: int = 1
    ) -> tuple[bool, int]:
        """
        Check if user exceeded rate limit.
        
        Returns:
            (is_allowed, remaining_requests)
        """
        async with get_db_session() as session:
            stmt = select(ChatUser).where(ChatUser.user_id == user_id)
            result = await session.execute(stmt)
            chat_user = result.scalar_one_or_none()
            
            if not chat_user:
                return True, max_requests - 1
            
            # Check if outside time window
            window_start = datetime.utcnow() - timedelta(minutes=window_minutes)
            
            if chat_user.last_request_time and chat_user.last_request_time < window_start:
                # Reset counter
                chat_user.requests_count = 0
                await session.commit()
                return True, max_requests - 1
            
            # Check limit
            remaining = max_requests - chat_user.requests_count
            is_allowed = chat_user.requests_count < max_requests
            
            return is_allowed, max(0, remaining)
