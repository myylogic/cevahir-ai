-- =============================================================================
-- PostgreSQL Schema for Cevahir Chatting Management System
-- =============================================================================
-- 
-- Endüstri Standartları:
-- - Proper indexing for performance
-- - Foreign key constraints for data integrity
-- - JSONB for flexible metadata storage
-- - Timestamps with timezone support
-- - Cascade delete for data consistency
--
-- Created: 2024
-- Database: PostgreSQL 12+
-- =============================================================================

-- Enable UUID extension (if needed)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- =============================================================================
-- Users Table
-- =============================================================================
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(36) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    google_id VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255),
    name VARCHAR(255),
    preferences JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Constraints
    CONSTRAINT check_email_format CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}$')
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
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Foreign Keys
    CONSTRAINT fk_sessions_user_id FOREIGN KEY (user_id) 
        REFERENCES users(user_id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT check_session_title_length CHECK (LENGTH(title) <= 255)
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
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Foreign Keys
    CONSTRAINT fk_messages_session_id FOREIGN KEY (session_id) 
        REFERENCES sessions(session_id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT check_message_role CHECK (role IN ('user', 'assistant', 'system')),
    CONSTRAINT check_message_content_not_empty CHECK (LENGTH(TRIM(content)) > 0)
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
    metadata JSONB,
    priority VARCHAR(20) NOT NULL DEFAULT 'medium',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Foreign Keys
    CONSTRAINT fk_user_memory_user_id FOREIGN KEY (user_id) 
        REFERENCES users(user_id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT check_memory_type CHECK (memory_type IN ('fact', 'preference', 'pattern', 'relationship', 'goal')),
    CONSTRAINT check_priority CHECK (priority IN ('high', 'medium', 'low')),
    CONSTRAINT check_memory_content_not_empty CHECK (LENGTH(TRIM(content)) > 0)
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
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP NOT NULL,
    
    -- Foreign Keys
    CONSTRAINT fk_conversation_summaries_session_id FOREIGN KEY (session_id) 
        REFERENCES sessions(session_id) ON DELETE CASCADE,
    
    -- Constraints
    CONSTRAINT check_summary_text_not_empty CHECK (LENGTH(TRIM(summary_text)) > 0),
    CONSTRAINT check_message_count_positive CHECK (message_count IS NULL OR message_count > 0)
);

-- Indexes for conversation_summaries
CREATE INDEX IF NOT EXISTS idx_conversation_summaries_session_id ON conversation_summaries(session_id);

-- =============================================================================
-- Triggers for updated_at timestamp
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for tables with updated_at
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_sessions_updated_at BEFORE UPDATE ON sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_memory_updated_at BEFORE UPDATE ON user_memory
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =============================================================================
-- Comments for Documentation
-- =============================================================================

COMMENT ON TABLE users IS 'User accounts with authentication information';
COMMENT ON TABLE sessions IS 'Conversation sessions for users';
COMMENT ON TABLE messages IS 'Individual messages in conversation sessions';
COMMENT ON TABLE user_memory IS 'Persistent user memory (facts, patterns, preferences)';
COMMENT ON TABLE conversation_summaries IS 'Summarized versions of long conversations';

COMMENT ON COLUMN users.user_id IS 'Unique user identifier (UUID)';
COMMENT ON COLUMN users.google_id IS 'Google OAuth ID (nullable)';
COMMENT ON COLUMN users.preferences IS 'User preferences stored as JSONB';

COMMENT ON COLUMN sessions.session_id IS 'Unique session identifier (UUID)';
COMMENT ON COLUMN sessions.metadata IS 'Session metadata stored as JSONB';

COMMENT ON COLUMN messages.role IS 'Message role: user, assistant, or system';
COMMENT ON COLUMN messages.metadata IS 'Message metadata (token count, model params, etc.) stored as JSONB';

COMMENT ON COLUMN user_memory.memory_type IS 'Type of memory: fact, preference, pattern, relationship, or goal';
COMMENT ON COLUMN user_memory.priority IS 'Memory priority: high, medium, or low';

-- =============================================================================
-- Initial Data (Optional)
-- =============================================================================

-- No initial data required - tables are ready for use

