-- =============================================================================
-- SQLite Schema for Cevahir Chatting Management System (Development)
-- =============================================================================
-- 
-- Endüstri Standartları:
-- - Proper indexing for performance
-- - Foreign key constraints for data integrity
-- - JSON support (TEXT with JSON serialization)
-- - Timestamps support
-- - Cascade delete for data consistency
--
-- Created: 2024
-- Database: SQLite 3.x
-- Usage: Development and testing only
-- =============================================================================

-- Enable foreign keys (SQLite specific)
PRAGMA foreign_keys = ON;

-- =============================================================================
-- Users Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    google_id VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255),
    name VARCHAR(255),
    preferences TEXT,  -- JSON stored as TEXT in SQLite
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Constraints
    CHECK (email LIKE '%@%.%')  -- Simple email format check
);

-- Indexes for users
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_google_id ON users(google_id) WHERE google_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- =============================================================================
-- Sessions Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    title VARCHAR(255),
    metadata TEXT,  -- JSON stored as TEXT
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Foreign Keys
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    
    -- Constraints
    CHECK (LENGTH(title) <= 255)
);

-- Indexes for sessions
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity);
CREATE INDEX IF NOT EXISTS idx_user_last_activity ON sessions(user_id, last_activity);

-- =============================================================================
-- Messages Table (Conversation History)
-- =============================================================================
CREATE TABLE IF NOT EXISTS messages (
    message_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    role VARCHAR(20) NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,  -- JSON stored as TEXT
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Foreign Keys
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    
    -- Constraints
    CHECK (role IN ('user', 'assistant', 'system')),
    CHECK (LENGTH(TRIM(content)) > 0)
);

-- Indexes for messages
CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at);
CREATE INDEX IF NOT EXISTS idx_session_created ON messages(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_session_role ON messages(session_id, role);

-- =============================================================================
-- User Memory Table (Persistent Facts, Patterns, Preferences)
-- =============================================================================
CREATE TABLE IF NOT EXISTS user_memory (
    memory_id VARCHAR(36) PRIMARY KEY,
    user_id VARCHAR(36) NOT NULL,
    memory_type VARCHAR(50) NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,  -- JSON stored as TEXT
    priority VARCHAR(20) NOT NULL DEFAULT 'medium',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Foreign Keys
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    
    -- Constraints
    CHECK (memory_type IN ('fact', 'preference', 'pattern', 'relationship', 'goal')),
    CHECK (priority IN ('high', 'medium', 'low')),
    CHECK (LENGTH(TRIM(content)) > 0)
);

-- Indexes for user_memory
CREATE INDEX IF NOT EXISTS idx_user_memory_user_id ON user_memory(user_id);
CREATE INDEX IF NOT EXISTS idx_user_memory_type ON user_memory(memory_type);
CREATE INDEX IF NOT EXISTS idx_user_memory_priority ON user_memory(priority);
CREATE INDEX IF NOT EXISTS idx_user_type_priority ON user_memory(user_id, memory_type, priority);

-- =============================================================================
-- Conversation Summaries Table (Optional - Performance Optimization)
-- =============================================================================
CREATE TABLE IF NOT EXISTS conversation_summaries (
    summary_id VARCHAR(36) PRIMARY KEY,
    session_id VARCHAR(36) NOT NULL,
    summary_text TEXT NOT NULL,
    message_count INTEGER,
    metadata TEXT,  -- JSON stored as TEXT
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Foreign Keys
    FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE,
    
    -- Constraints
    CHECK (LENGTH(TRIM(summary_text)) > 0),
    CHECK (message_count IS NULL OR message_count > 0)
);

-- Indexes for conversation_summaries
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_session_id ON conversation_summaries(session_id);

-- =============================================================================
-- Triggers for updated_at timestamp (SQLite specific)
-- =============================================================================

-- Function equivalent: SQLite triggers for updated_at
CREATE TRIGGER IF NOT EXISTS update_users_updated_at
    AFTER UPDATE ON users
    FOR EACH ROW
    WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE users SET updated_at = CURRENT_TIMESTAMP WHERE user_id = NEW.user_id;
END;

CREATE TRIGGER IF NOT EXISTS update_sessions_updated_at
    AFTER UPDATE ON sessions
    FOR EACH ROW
    WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE session_id = NEW.session_id;
END;

CREATE TRIGGER IF NOT EXISTS update_user_memory_updated_at
    AFTER UPDATE ON user_memory
    FOR EACH ROW
    WHEN NEW.updated_at = OLD.updated_at
BEGIN
    UPDATE user_memory SET updated_at = CURRENT_TIMESTAMP WHERE memory_id = NEW.memory_id;
END;

-- =============================================================================
-- Comments for Documentation (SQLite doesn't support COMMENT, using -- instead)
-- =============================================================================

-- users: User accounts with authentication information
-- sessions: Conversation sessions for users
-- messages: Individual messages in conversation sessions
-- user_memory: Persistent user memory (facts, patterns, preferences)
-- conversation_summaries: Summarized versions of long conversations

