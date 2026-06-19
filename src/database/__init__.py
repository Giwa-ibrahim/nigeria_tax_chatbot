"""
Database module for chat state management.
"""
from src.database.models import (
    Base,
    ChatSession,
    ChatMessage,
    ChatSummary,
    ChatUser
)
from src.database.connection import (
    get_async_engine,
    get_session_factory,
    get_db_session,
    init_database,
    close_database,
    health_check
)
from src.database.repository import (
    ChatSessionRepository,
    ChatMessageRepository,
    ChatUserRepository
)
from src.database.chat_manager import ChatManager

__all__ = [
    # Models
    "Base",
    "ChatSession",
    "ChatMessage",
    "ChatSummary",
    "ChatUser",
    # Connection
    "get_async_engine",
    "get_session_factory",
    "get_db_session",
    "init_database",
    "close_database",
    "health_check",
    # Session Store
    "SessionStore",
    # Repositories
    "ChatSessionRepository",
    "ChatMessageRepository",
    "ChatUserRepository",
    # Utilities
    "ChatManager",
]
