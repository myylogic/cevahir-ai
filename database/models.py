# -*- coding: utf-8 -*-
"""
SQLAlchemy Database Models
==========================

PostgreSQL-optimized models with JSONB support.
Endüstri Standartları: Proper indexing, foreign keys, constraints.

SOLID Principles:
- Single Responsibility: Each model represents one entity
- Open/Closed: Extensible via inheritance
"""

import json
from typing import Optional, Dict, Any
from datetime import datetime

from sqlalchemy import (
    Column, String, Text, Integer, DateTime, ForeignKey, Index,
    UniqueConstraint, CheckConstraint
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import TypeDecorator, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship

Base = declarative_base()


# =============================================================================
# Custom Types
# =============================================================================

class JSONType(TypeDecorator):
    """
    JSON/JSONB type that works on all databases.
    
    PostgreSQL: JSONB (efficient, indexed)
    MySQL: JSON
    SQLite: TEXT (JSON serialized)
    
    Endüstri Standardı: Database-agnostic JSON handling
    """
    impl = Text
    cache_ok = True
    
    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(JSONB())
        elif dialect.name == 'mysql':
            return dialect.type_descriptor(JSON())
        else:
            return dialect.type_descriptor(Text())
    
    def process_bind_param(self, value, dialect):
        """Serialize JSON to string for database"""
        if value is None:
            return value
        if dialect.name == 'postgresql':
            # PostgreSQL JSONB accepts dict/list directly
            return value
        return json.dumps(value, ensure_ascii=False)
    
    def process_result_value(self, value, dialect):
        """Deserialize JSON from database"""
        if value is None:
            return value
        if isinstance(value, (dict, list)):
            # Already parsed (PostgreSQL JSONB)
            return value
        return json.loads(value)


# =============================================================================
# Models
# =============================================================================

class User(Base):
    """
    User model.
    
    Stores user account information including authentication.
    """
    __tablename__ = "users"
    
    # Primary Key
    user_id = Column(String(36), primary_key=True, comment="Unique user identifier (UUID)")
    
    # Authentication
    email = Column(String(255), unique=True, nullable=False, index=True, comment="User email address")
    google_id = Column(String(255), unique=True, nullable=True, index=True, comment="Google OAuth ID")
    password_hash = Column(String(255), nullable=True, comment="Password hash (bcrypt)")
    
    # User Information
    name = Column(String(255), nullable=True, comment="User display name")
    preferences = Column(JSONType, nullable=True, comment="User preferences (JSON)")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, comment="Account creation timestamp")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False, comment="Last update timestamp")
    
    # Relationships
    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("UserMemory", back_populates="user", cascade="all, delete-orphan")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}$'", name="check_email_format"),
    )
    
    def __repr__(self) -> str:
        return f"<User(user_id={self.user_id}, email={self.email})>"


class Session(Base):
    """
    Session model.
    
    Represents a conversation session for a user.
    """
    __tablename__ = "sessions"
    
    # Primary Key
    session_id = Column(String(36), primary_key=True, comment="Unique session identifier (UUID)")
    
    # Foreign Key
    user_id = Column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Owner user ID"
    )
    
    # Session Information
    title = Column(String(255), nullable=True, comment="Session title (auto-generated or user-set)")
    metadata = Column(JSONType, nullable=True, comment="Session metadata (JSON)")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, comment="Session creation timestamp")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False, comment="Last update timestamp")
    last_activity = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True, comment="Last activity timestamp")
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    messages = relationship("Message", back_populates="session", cascade="all, delete-orphan", order_by="Message.created_at")
    summaries = relationship("ConversationSummary", back_populates="session", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_user_last_activity', 'user_id', 'last_activity'),
    )
    
    def __repr__(self) -> str:
        return f"<Session(session_id={self.session_id}, user_id={self.user_id}, title={self.title})>"


class Message(Base):
    """
    Message model.
    
    Stores individual messages in a conversation session.
    """
    __tablename__ = "messages"
    
    # Primary Key
    message_id = Column(String(36), primary_key=True, comment="Unique message identifier (UUID)")
    
    # Foreign Key
    session_id = Column(
        String(36),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent session ID"
    )
    
    # Message Content
    role = Column(String(20), nullable=False, comment="Message role: 'user' or 'assistant'")
    content = Column(Text, nullable=False, comment="Message content")
    metadata = Column(JSONType, nullable=True, comment="Message metadata (token count, model params, etc.)")
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True, comment="Message creation timestamp")
    
    # Relationships
    session = relationship("Session", back_populates="messages")
    
    # Constraints and Indexes
    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="check_role"),
        Index('idx_session_created', 'session_id', 'created_at'),
        Index('idx_session_role', 'session_id', 'role'),
    )
    
    def __repr__(self) -> str:
        role_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<Message(message_id={self.message_id}, role={self.role}, content='{role_preview}')>"


class UserMemory(Base):
    """
    User memory model.
    
    Stores persistent user memory (facts, preferences, patterns, relationships).
    """
    __tablename__ = "user_memory"
    
    # Primary Key
    memory_id = Column(String(36), primary_key=True, comment="Unique memory identifier (UUID)")
    
    # Foreign Key
    user_id = Column(
        String(36),
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Owner user ID"
    )
    
    # Memory Content
    memory_type = Column(
        String(50),
        nullable=False,
        index=True,
        comment="Memory type: 'fact', 'preference', 'pattern', 'relationship', 'goal'"
    )
    content = Column(Text, nullable=False, comment="Memory content")
    metadata = Column(JSONType, nullable=True, comment="Memory metadata (JSON)")
    priority = Column(
        String(20),
        nullable=False,
        default="medium",
        index=True,
        comment="Priority: 'high', 'medium', 'low'"
    )
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, comment="Memory creation timestamp")
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False, comment="Last update timestamp")
    
    # Relationships
    user = relationship("User", back_populates="memories")
    
    # Constraints
    __table_args__ = (
        CheckConstraint("memory_type IN ('fact', 'preference', 'pattern', 'relationship', 'goal')", name="check_memory_type"),
        CheckConstraint("priority IN ('high', 'medium', 'low')", name="check_priority"),
        Index('idx_user_type_priority', 'user_id', 'memory_type', 'priority'),
    )
    
    def __repr__(self) -> str:
        content_preview = self.content[:50] + "..." if len(self.content) > 50 else self.content
        return f"<UserMemory(memory_id={self.memory_id}, type={self.memory_type}, priority={self.priority}, content='{content_preview}')>"


class ConversationSummary(Base):
    """
    Conversation summary model.
    
    Stores summarized versions of long conversations for context optimization.
    """
    __tablename__ = "conversation_summaries"
    
    # Primary Key
    summary_id = Column(String(36), primary_key=True, comment="Unique summary identifier (UUID)")
    
    # Foreign Key
    session_id = Column(
        String(36),
        ForeignKey("sessions.session_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Parent session ID"
    )
    
    # Summary Content
    summary_text = Column(Text, nullable=False, comment="Summary text")
    message_count = Column(Integer, nullable=True, comment="Number of messages summarized")
    metadata = Column(JSONType, nullable=True, comment="Summary metadata (JSON)")
    
    # Timestamp
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, comment="Summary creation timestamp")
    
    # Relationships
    session = relationship("Session", back_populates="summaries")
    
    def __repr__(self) -> str:
        summary_preview = self.summary_text[:50] + "..." if len(self.summary_text) > 50 else self.summary_text
        return f"<ConversationSummary(summary_id={self.summary_id}, session_id={self.session_id}, summary='{summary_preview}')>"

