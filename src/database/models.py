"""
SQLAlchemy models for chat storage.
"""
from datetime import datetime
from typing import Optional
from uuid import uuid4
from sqlalchemy import (
    Column, String, Integer, DateTime, Text, JSON, 
    ForeignKey, Index
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class ChatSession(Base):
    """
    Main chat session table.
    Stores conversation metadata and current agent state.
    """
    __tablename__ = "chat_sessions"
    
    # Primary Key (UUID4)
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()), index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # User Info
    user_id = Column(String(255), nullable=True, index=True)
    
    # Session State
    status = Column(String(50), default="active", nullable=False)  # active, archived
    message_count = Column(Integer, default=0, nullable=False)
    
    # Agent State (stores latest conversation state)
    current_state = Column(JSON, nullable=True)  # Latest agent state
    
    # Extra metadata (renamed from 'metadata' - reserved word in SQLAlchemy)
    extra_metadata = Column(JSON, nullable=True)
    
    # Relationships
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan")
    summaries = relationship("ChatSummary", back_populates="session", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_chat_session_user_id', 'user_id'),
        Index('idx_chat_session_created_at', 'created_at'),
        Index('idx_chat_session_status', 'status'),
    )
    
    def __repr__(self):
        return f"<ChatSession(id={self.id}, user={self.user_id}, messages={self.message_count})>"


class ChatMessage(Base):
    """
    Individual messages within a chat session.
    Stores user queries and agent responses.
    """
    __tablename__ = "chat_messages"
    
    # Primary Key (UUID4)
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()), index=True)
    
    # Foreign Key
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Message Content
    role = Column(String(50), nullable=False)  # user, assistant
    content = Column(Text, nullable=False)
    
    # Agent Info
    agent_type = Column(String(100), nullable=True)  # tax_policy, paye, financial, combined
    
    # Token Tracking
    tokens_used = Column(Integer, nullable=True)
    
    # Relationships
    session = relationship("ChatSession", back_populates="messages")
    
    # Indexes
    __table_args__ = (
        Index('idx_chat_message_session_id', 'session_id'),
        Index('idx_chat_message_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<ChatMessage(id={self.id}, role={self.role}, session={self.session_id})>"


class ChatSummary(Base):
    """
    Compressed conversation summaries for context management.
    Reduces token usage by storing periodic summaries.
    """
    __tablename__ = "chat_summaries"
    
    # Primary Key (UUID4)
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()), index=True)
    
    # Foreign Key
    session_id = Column(String(36), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Summary Content
    summary_text = Column(Text, nullable=False)  # Compressed conversation
    message_range = Column(String(100), nullable=True)  # e.g., "1-20"
    
    # Relationships
    session = relationship("ChatSession", back_populates="summaries")
    
    # Indexes
    __table_args__ = (
        Index('idx_chat_summary_session_id', 'session_id'),
        Index('idx_chat_summary_created_at', 'created_at'),
    )
    
    def __repr__(self):
        return f"<ChatSummary(id={self.id}, session={self.session_id}, range={self.message_range})>"


class ChatUser(Base):
    """
    User tracking for analytics and rate limiting.
    """
    __tablename__ = "chat_users"
    
    # Primary Key (UUID4)
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()), index=True)
    
    # User Identification
    user_id = Column(String(255), nullable=False, unique=True, index=True)
    
    # Activity Tracking
    last_activity = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Rate Limiting
    requests_count = Column(Integer, default=0, nullable=False)
    last_request_time = Column(DateTime, nullable=True)
    
    # Indexes
    __table_args__ = (
        Index('idx_chat_user_user_id', 'user_id'),
        Index('idx_chat_user_last_activity', 'last_activity'),
    )
    
    def __repr__(self):
        return f"<ChatUser(id={self.id}, user_id={self.user_id})>"


class UserPreference(Base):
    """
    Learned user preferences across sessions
    Helps agent get smarter over time
    """
    __tablename__ = "user_preferences"
    
    # Primary Key (UUID4)
    id = Column(String(36), primary_key=True, default=lambda: str(uuid4()), index=True)
    
    # User Identification
    user_id = Column(String(255), nullable=False, index=True)
    
    # Learned preferences
    preferred_communication_style = Column(String(50), nullable=True)  # "concise" | "detailed" | "balanced"
    topic_interests = Column(JSON, default=dict)  # {"paye": 15, "tax_policy": 8}
    calculation_defaults = Column(JSON, default=dict)  # Preferred PAYE inputs
    
    # Metadata
    total_sessions = Column(Integer, default=0)
    last_updated = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index("idx_user_pref_user_updated", "user_id", "last_updated"),
    )
    
    def __repr__(self):
        return f"<UserPreference(user_id={self.user_id}, style={self.preferred_communication_style})>"

