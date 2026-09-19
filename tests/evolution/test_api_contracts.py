"""API and SQLite integration without loading or training a language model."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

deps = Path(__file__).resolve().parents[2] / ".evolution" / "deps"
if deps.exists():
    sys.path.insert(0, str(deps))
pytest.importorskip("flask")
pytest.importorskip("sqlalchemy")

from api.app_factory import create_app
from database.models import Base, Session as ChatSession, Message
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def test_two_app_factories_have_independent_registered_routes():
    model = SimpleNamespace()
    config = {"TESTING": True, "RATELIMIT_ENABLED": False, "ENABLE_METRICS": False}
    apps = [create_app(config, cevahir=model, chatting_manager=SimpleNamespace(), initialize_db=False) for _ in range(2)]
    for app in apps:
        assert app.test_client().get("/").status_code == 200
        assert "/api/v3/chat/messages" in {rule.rule for rule in app.url_map.iter_rules()}
    assert apps[0].blueprints["v3"] is not apps[1].blueprints["v3"]


def test_sql_metadata_column_keeps_name_and_legacy_instance_access():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    assert "metadata" in ChatSession.__table__.columns
    assert "metadata" in Message.__table__.columns
    instance = ChatSession(metadata={"language": "tr"})
    assert instance.metadata_json == {"language": "tr"}
    instance.metadata = {"language": "en"}
    assert instance.metadata_json == {"language": "en"}
    assert hasattr(Base.metadata, "create_all")


def test_authenticated_sessions_use_token_identity_and_app_specific_keys():
    from api.security.jwt import create_access_token, create_refresh_token
    from datetime import timedelta
    class Sessions:
        def list_sessions(self, user_id, limit=None):
            return [{"session_id": user_id + "-session", "user_id": user_id}]
    apps = [create_app({"TESTING": True, "DEBUG": True, "RATELIMIT_ENABLED": False,
                        "ENABLE_METRICS": False, "JWT_SECRET_KEY": key * 40},
                       cevahir=SimpleNamespace(), chatting_manager=Sessions(), initialize_db=False)
            for key in ("a", "b")]
    with apps[0].app_context():
        alice = create_access_token("alice")
        bob = create_access_token("bob")
        expired = create_access_token("alice", expires_delta=timedelta(seconds=-1))
        refresh = create_refresh_token("alice")
        with pytest.raises(ValueError):
            create_access_token("alice", additional_claims={"sub": "bob"})
    client = apps[0].test_client()
    for user, token in (("alice", alice), ("bob", bob)):
        response = client.get("/api/v3/sessions", headers={"Authorization": "Bearer " + token, "X-User-ID": "spoofed"})
        assert response.status_code == 200
        assert response.json["data"]["sessions"][0]["user_id"] == user
    for token in ("invalid", expired, refresh):
        assert client.get("/api/v3/sessions", headers={"Authorization": "Bearer " + token, "X-User-ID": "bob"}).status_code == 401
    assert apps[1].test_client().get("/api/v3/sessions", headers={"Authorization": "Bearer " + alice}).status_code == 401


def test_jwt_requires_expiration_and_rejects_default_signing_key():
    from flask import Flask
    import jwt
    from api.security.jwt import verify_token, create_access_token
    app = Flask(__name__)
    app.config["JWT_SECRET_KEY"] = "s" * 40
    with app.app_context():
        token = jwt.encode({"sub": "alice", "iat": 1, "type": "access"}, "s" * 40, algorithm="HS256")
        assert verify_token(token) is None
        app.config["JWT_SECRET_KEY"] = "change-me"
        with pytest.raises(RuntimeError, match="JWT_SECRET_KEY"):
            create_access_token("alice")


def test_real_sqlite_session_ownership_and_metadata_persistence(tmp_path, monkeypatch):
    from sqlalchemy.orm import scoped_session, sessionmaker
    from database.connection import db
    from database import UnitOfWork
    from database.models import User
    from chatting_management import ChattingManager, ChattingConfig
    from chatting_management.exceptions import SessionAccessDeniedError
    from api.security.jwt import create_access_token
    engine = create_engine("sqlite:///" + str(tmp_path / "ownership.db"))
    Base.metadata.create_all(engine)
    factory = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setattr(db, "get_session_factory", lambda: factory)
    try:
        with UnitOfWork() as uow:
            for name in ("alice", "bob"):
                uow.users.create(User(user_id=name, email=name + "@example.com"))
        class Model:
            calls = 0
            def process(self, **kwargs):
                self.calls += 1
                raise AssertionError("Unauthorized requests must not invoke the model")
        model = Model()
        manager = ChattingManager(ChattingConfig(enable_user_memory=False, enable_semantic_search=False), model)
        sid = manager.create_session("alice", title="private")["session_id"]
        manager.conversation_manager.add_message(session_id=sid, role="user", content="private content")
        manager.session_storage.update_session_metadata(sid, {"first": 1})
        manager.session_storage.update_session_metadata(sid, {"second": 2})
        assert manager.session_storage.get_session(sid).metadata == {"first": 1, "second": 2}
        manager.user_storage.update_user_preferences("alice", {"language": "tr"})
        manager.user_storage.update_user_preferences("alice", {"theme": "dark"})
        with UnitOfWork() as uow:
            assert uow.users.get_by_id("alice").preferences == {"language": "tr", "theme": "dark"}
        from chatting_management.storage.memory_storage import MemoryStorage
        memory_storage = MemoryStorage()
        memory = memory_storage.add_memory(user_id="alice", memory_type="preference", content="Turkish", metadata={"a": 1})
        memory_storage.update_memory(memory.memory_id, metadata={"b": 2})
        with UnitOfWork() as uow:
            assert uow.user_memories.get_by_id(memory.memory_id).metadata == {"a": 1, "b": 2}
        with pytest.raises(SessionAccessDeniedError):
            manager.session_manager.get_session(sid, user_id="")
        app = create_app({"TESTING": True, "DEBUG": False, "RATELIMIT_ENABLED": False,
                          "ENABLE_METRICS": False, "JWT_SECRET_KEY": "ownership-test-" * 4},
                         cevahir=model, chatting_manager=manager, initialize_db=False)
        with app.app_context():
            tokens = {user: create_access_token(user) for user in ("alice", "bob")}
        client = app.test_client()
        headers = lambda user: {"Authorization": "Bearer " + tokens[user]}
        assert client.get("/api/v3/chat/messages", query_string={"session_id": sid}, headers=headers("alice")).status_code == 200
        denied = client.get("/api/v3/chat/messages", query_string={"session_id": sid}, headers=headers("bob"))
        assert denied.status_code == 403 and "private content" not in denied.get_data(as_text=True)
        assert client.post("/api/v3/chat/messages", json={"session_id": sid, "message": "injected"}, headers=headers("bob")).status_code == 403
        assert client.get("/api/v3/chat/messages", query_string={"session_id": "missing"}, headers=headers("alice")).status_code == 404
        sessions = client.get("/api/v3/sessions", headers=headers("bob"))
        assert sessions.status_code == 200 and sessions.json["data"]["sessions"] == []
        assert model.calls == 0
        assert len(manager.get_conversation_history(sid, "alice")) == 1
        from datetime import datetime, timedelta
        with UnitOfWork() as uow:
            uow.messages.get_by_session_id(sid)[0].created_at = datetime(2020, 1, 1)
            for i in range(1, 4):
                uow.messages.create(Message(message_id="recent-" + str(i), session_id=sid,
                                             role="user", content=str(i), created_at=datetime(2020, 1, 1) + timedelta(seconds=i)))
        monkeypatch.setattr(manager.context_builder, "_estimate_tokens", lambda history: len(history))
        history = manager.context_builder._get_recent_history(sid, max_tokens=2)
        assert [item["content"] for item in history] == ["2", "3"]
    finally:
        factory.remove()
        engine.dispose()
