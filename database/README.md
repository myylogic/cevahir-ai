# 🗄️ Database Module

PostgreSQL Database Management for Cevahir Chatting Management System.

## 📋 Overview

Endüstri standartlarında, SOLID prensiplerine uygun database modülü.

**Features:**
- ✅ PostgreSQL optimized (JSONB support, vector extension ready)
- ✅ SOLID Principles (Repository Pattern, Dependency Inversion)
- ✅ Connection pooling (efficient resource management)
- ✅ Transaction management (Unit of Work Pattern)
- ✅ Type-safe models (SQLAlchemy ORM)
- ✅ Comprehensive error handling
- ✅ Extensive logging

## 🏗️ Architecture

### SOLID Principles

- **Single Responsibility:** Each repository handles one entity type
- **Open/Closed:** Extensible via inheritance and interfaces
- **Liskov Substitution:** Repository interfaces are substitutable
- **Interface Segregation:** Focused, small interfaces
- **Dependency Inversion:** High-level modules depend on interfaces

### Design Patterns

- **Repository Pattern:** Data access abstraction
- **Unit of Work Pattern:** Transaction management
- **Singleton Pattern:** Connection pool management
- **Factory Pattern:** Connection factory

## 📁 Module Structure

```
database/
├── __init__.py              # Public API
├── config.py                # Configuration (PostgreSQL)
├── connection.py            # Connection management (Singleton)
├── models.py                # SQLAlchemy models
├── exceptions.py            # Custom exceptions
├── unit_of_work.py          # Unit of Work implementation
│
├── interfaces/              # Repository interfaces (Protocols)
│   ├── repository.py
│   └── unit_of_work.py
│
├── repositories/            # Repository implementations
│   ├── base_repository.py
│   ├── user_repository.py
│   ├── session_repository.py
│   ├── message_repository.py
│   └── user_memory_repository.py
│
├── schemas/                 # SQL schema files
│   └── schema_postgresql.sql
│
├── migrations/              # Migration scripts (future)
│
└── utils/                   # Helper functions
    └── helpers.py
```

## 🚀 Quick Start

### 1. Configuration

Set environment variables:

```bash
# PostgreSQL Configuration
export DB_TYPE=postgresql
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=cevahir
export POSTGRES_USER=cevahir
export POSTGRES_PASSWORD=your_password

# Connection Pool
export DB_POOL_SIZE=10
export DB_MAX_OVERFLOW=20
```

### 2. Initialize Database

```python
from database import db, Base
from database.models import User, Session, Message

# Initialize connection
db.initialize()

# Create tables (development only)
Base.metadata.create_all(db.get_engine())
```

### 3. Use Repository

```python
from database import db
from database.models import User
from database.repositories.user_repository import UserRepository

# Using repository
with db.get_session() as session:
    user_repo = UserRepository(session)
    user = user_repo.get_by_email("user@example.com")
```

### 4. Use Unit of Work

```python
from database.unit_of_work import UnitOfWork
from database.utils.helpers import generate_uuid

# Multiple operations in single transaction
with UnitOfWork() as uow:
    # Create user
    user = User(
        user_id=generate_uuid(),
        email="user@example.com",
        name="Test User"
    )
    uow.users.create(user)
    
    # Create session
    session = Session(
        session_id=generate_uuid(),
        user_id=user.user_id,
        title="Test Session"
    )
    uow.sessions.create(session)
    
    # Commit (automatic on context exit if no exception)
    uow.commit()
```

## 📊 Models

### User
- `user_id` (PK)
- `email` (unique)
- `google_id` (unique, nullable)
- `preferences` (JSONB)

### Session
- `session_id` (PK)
- `user_id` (FK → users)
- `title`
- `metadata` (JSONB)

### Message
- `message_id` (PK)
- `session_id` (FK → sessions)
- `role` (user/assistant)
- `content`
- `metadata` (JSONB)

### UserMemory
- `memory_id` (PK)
- `user_id` (FK → users)
- `memory_type` (fact/preference/pattern/relationship)
- `content`
- `priority` (high/medium/low)

## 🔌 Integration

### Flask

```python
from database import db

def create_app():
    app = Flask(__name__)
    db.initialize()
    app.db = db
    return app
```

### ChattingManagement

```python
from database import db
from database.unit_of_work import UnitOfWork

class ChattingManager:
    def send_message(self, user_id, session_id, message):
        with UnitOfWork() as uow:
            # Database operations
            ...
```

## 📝 Requirements

```
sqlalchemy>=2.0.0
psycopg2-binary>=2.9.0  # PostgreSQL driver
```

## 🔐 Security

- ✅ Password hashing (bcrypt)
- ✅ SQL injection prevention (parameterized queries)
- ✅ Connection pooling (resource management)
- ✅ Transaction isolation

## 📚 Documentation

For detailed documentation, see:
- `docs/DATABASE_MIMARI_VE_KURULUM_PLANI.md` - Architecture and setup
- `docs/CHATTING_MANAGEMENT_BETA_PLAN.md` - Integration guide

