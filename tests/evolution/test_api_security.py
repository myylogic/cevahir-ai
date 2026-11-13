"""Security contracts of the HTTP API: identity source, configuration modes, CORS,
health-check disclosure and the development-server defaults.

No language model is loaded; the data-access case uses a temporary SQLite file.
"""
import logging
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

deps = Path(__file__).resolve().parents[2] / ".evolution" / "deps"
if deps.exists():
    sys.path.insert(0, str(deps))
for _module in ("flask", "flask_cors", "flask_limiter", "jwt", "sqlalchemy", "bcrypt"):
    pytest.importorskip(_module)

from api import api_config, app_factory
from api.api_config import ConfigurationError, get_config
from api.security.jwt import create_access_token

ENVIRONMENT = (
    "FLASK_ENV", "CEVAHIR_TRUST_X_USER_ID", "CORS_ORIGINS", "CORS_METHODS", "CORS_HEADERS",
    "RATELIMIT_ENABLED", "SECRET_KEY", "FLASK_SECRET_KEY", "JWT_SECRET_KEY", "PRODUCTION_SECRET_KEY",
    "DEVELOPMENT_SECRET_KEY", "TESTING_SECRET_KEY", "CEVAHIR_API_HOST", "CEVAHIR_API_DEBUG", "PORT",
)
JWT_KEY = "jwt-signing-key-used-only-by-these-tests"
FLASK_KEY = "flask-secret-key-used-only-by-these-tests"
PRIVATE = "alice's private note"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    # Another test module may have disabled logging process-wide; these tests read log records.
    disabled = logging.root.manager.disable
    logging.disable(logging.NOTSET)
    yield
    logging.disable(disabled)


def production_environment(monkeypatch, **overrides):
    values = {"FLASK_ENV": "production", "PRODUCTION_SECRET_KEY": FLASK_KEY, "JWT_SECRET_KEY": JWT_KEY}
    values.update(overrides)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


class Conversations:
    """ChattingManager stand-in that records which identity reached it."""

    def __init__(self):
        self.calls = []

    def list_sessions(self, user_id, limit=None):
        self.calls.append(("list_sessions", user_id))
        return [{"session_id": user_id + "-s", "user_id": user_id}]


def make_app(overrides=None, manager=None):
    config = {"ENABLE_METRICS": False}
    config.update(overrides or {})
    return app_factory.create_app(config, cevahir=SimpleNamespace(),
                                  chatting_manager=manager or Conversations(), initialize_db=False)


def preflight(client, origin, path="/api/v3/sessions"):
    return client.open(path, method="OPTIONS", headers={
        "Origin": origin, "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "Authorization"})


# Identity -------------------------------------------------------------------

def test_forged_user_header_cannot_read_another_users_conversation(tmp_path, monkeypatch):
    from sqlalchemy import create_engine
    from sqlalchemy.orm import scoped_session, sessionmaker
    from database import UnitOfWork
    from database.connection import db
    from database.models import Base, User
    from chatting_management import ChattingManager, ChattingConfig

    class NoModel:
        def process(self, **kwargs):
            raise AssertionError("Unauthenticated requests must not reach the model")

    engine = create_engine("sqlite:///" + str(tmp_path / "identity.db"))
    Base.metadata.create_all(engine)
    factory = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
    monkeypatch.setattr(db, "get_session_factory", lambda: factory)
    try:
        with UnitOfWork() as uow:
            for name in ("alice", "mallory"):
                uow.users.create(User(user_id=name, email=name + "@example.com"))
        manager = ChattingManager(ChattingConfig(enable_user_memory=False, enable_semantic_search=False), NoModel())
        sid = manager.create_session("alice", title="private")["session_id"]
        manager.conversation_manager.add_message(session_id=sid, role="user", content=PRIVATE)
        # DEBUG=True used to be enough to trust X-User-ID; it no longer is.
        app = make_app({"JWT_SECRET_KEY": JWT_KEY, "DEBUG": True}, manager=manager)
        client = app.test_client()
        forged = {"X-User-ID": "alice"}
        for response in (client.get("/api/v3/chat/messages", query_string={"session_id": sid}, headers=forged),
                         client.post("/api/v3/chat/messages", json={"session_id": sid, "message": "hi"}, headers=forged),
                         client.get("/api/v3/sessions", headers=forged),
                         client.get("/api/v3/users/me", headers=forged)):
            assert response.status_code == 401
            assert response.json["error"]["code"] == "AUTHENTICATION_ERROR"
            assert PRIVATE not in response.get_data(as_text=True)
        with app.app_context():
            tokens = {user: create_access_token(user) for user in ("alice", "mallory")}
        # A valid token for mallory stays mallory even with alice's id in the header.
        denied = client.get("/api/v3/chat/messages", query_string={"session_id": sid},
                            headers={"Authorization": "Bearer " + tokens["mallory"], "X-User-ID": "alice"})
        assert denied.status_code == 403 and PRIVATE not in denied.get_data(as_text=True)
        owner = client.get("/api/v3/chat/messages", query_string={"session_id": sid},
                           headers={"Authorization": "Bearer " + tokens["alice"]})
        assert owner.status_code == 200 and PRIVATE in owner.get_data(as_text=True)
        assert len(manager.get_conversation_history(sid, "alice")) == 1
    finally:
        factory.remove()
        engine.dispose()


@pytest.mark.parametrize("mode", ["development", "testing"])
def test_user_header_needs_the_opt_in_and_an_explicit_development_mode(monkeypatch, caplog, mode):
    monkeypatch.setenv("FLASK_ENV", mode)
    manager = Conversations()
    assert make_app(manager=manager).test_client().get(
        "/api/v3/sessions", headers={"X-User-ID": "alice"}).status_code == 401
    assert manager.calls == []

    monkeypatch.setenv("CEVAHIR_TRUST_X_USER_ID", "1")
    app = make_app(manager=manager)
    assert app.config["TRUST_X_USER_ID"] is True and "X-User-ID" in app.config["CORS_HEADERS"]
    with caplog.at_level(logging.WARNING, logger="api.middleware.auth"):
        response = app.test_client().get("/api/v3/sessions", headers={"X-User-ID": "alice"})
    assert response.status_code == 200
    assert response.json["data"]["sessions"][0]["user_id"] == "alice"
    assert manager.calls == [("list_sessions", "alice")]
    assert any(record.levelno == logging.WARNING and "X-User-ID" in record.getMessage()
               for record in caplog.records if record.name == "api.middleware.auth")


@pytest.mark.parametrize("mode", ["", "production"])
def test_user_header_opt_in_is_refused_outside_development_and_testing(monkeypatch, mode):
    production_environment(monkeypatch, FLASK_ENV=mode, CEVAHIR_TRUST_X_USER_ID="1")
    with pytest.raises(ConfigurationError, match="CEVAHIR_TRUST_X_USER_ID"):
        get_config()


def test_production_app_ignores_a_programmatic_user_header_override(monkeypatch):
    production_environment(monkeypatch)
    manager = Conversations()
    app = make_app({"TRUST_X_USER_ID": True}, manager=manager)
    assert app.config["API_ENV"] == "production"
    assert app.test_client().get("/api/v3/sessions", headers={"X-User-ID": "alice"}).status_code == 401
    assert manager.calls == []


# Configuration modes --------------------------------------------------------

def test_unset_flask_env_selects_the_safe_default(monkeypatch):
    config = get_config()
    assert type(config) is api_config.DefaultConfig and config.API_ENV == "default"
    assert config.DEBUG is False and config.RATELIMIT_ENABLED is True and config.TRUST_X_USER_ID is False
    assert config.CORS_ORIGINS == list(api_config.LOCALHOST_CORS_ORIGINS)
    assert "X-User-ID" not in config.CORS_HEADERS
    app = make_app({"RATELIMIT_DEFAULT": "2 per minute"})
    assert app.debug is False
    client = app.test_client()
    assert [client.get("/api/v3/health").status_code for _ in range(3)] == [200, 200, 429]


def test_explicit_development_mode_keeps_debug_but_not_wildcard_cors(monkeypatch):
    monkeypatch.setenv("FLASK_ENV", "development")
    config = get_config()
    assert type(config) is api_config.DevelopmentConfig
    assert config.DEBUG is True and config.RATELIMIT_ENABLED is False and config.TRUST_X_USER_ID is False
    assert config.CORS_ORIGINS == list(api_config.LOCALHOST_CORS_ORIGINS)


@pytest.mark.parametrize("name, value", [("FLASK_ENV", "prodution"), ("RATELIMIT_ENABLED", "maybe"),
                                         ("CEVAHIR_TRUST_X_USER_ID", "enabled")])
def test_unrecognised_mode_or_flag_values_are_errors(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ConfigurationError, match=name):
        get_config()


STRONG = "a-production-flask-secret-of-adequate-length"


@pytest.mark.parametrize("environment, message", [
    # The audit's reproduction: FLASK_ENV=production PRODUCTION_SECRET_KEY=x, no JWT_SECRET_KEY.
    ({"PRODUCTION_SECRET_KEY": "x", "JWT_SECRET_KEY": ""}, "JWT_SECRET_KEY is not set"),
    ({"PRODUCTION_SECRET_KEY": "x"}, "PRODUCTION_SECRET_KEY is shorter than 32"),
    ({"JWT_SECRET_KEY": "change-me-in-production"}, "JWT_SECRET_KEY uses the placeholder"),
    ({"JWT_SECRET_KEY": "", "SECRET_KEY": "change-me-in-production"}, "JWT_SECRET_KEY is not set"),
    ({"JWT_SECRET_KEY": "", "FLASK_SECRET_KEY": STRONG}, "JWT_SECRET_KEY is not set"),
    ({"PRODUCTION_SECRET_KEY": "change-me-in-production"}, "PRODUCTION_SECRET_KEY uses the placeholder"),
    ({"PRODUCTION_SECRET_KEY": ""}, "PRODUCTION_SECRET_KEY is not set"),
    ({"JWT_SECRET_KEY": "short"}, "JWT_SECRET_KEY is shorter than 32"),
    ({"PRODUCTION_SECRET_KEY": STRONG, "JWT_SECRET_KEY": STRONG}, "must differ"),
])
def test_production_configuration_refuses_placeholder_and_weak_secrets(monkeypatch, environment, message):
    production_environment(monkeypatch, **environment)
    with pytest.raises(ConfigurationError, match=message):
        get_config()
    with pytest.raises(ConfigurationError, match=message):
        make_app()


def test_production_configuration_with_real_secrets(monkeypatch):
    production_environment(monkeypatch)
    config = get_config()
    assert type(config) is api_config.ProductionConfig
    assert config.SECRET_KEY == FLASK_KEY and config.JWT_SECRET_KEY == JWT_KEY
    assert config.DEBUG is False and config.RATELIMIT_ENABLED is True and config.TRUST_X_USER_ID is False
    assert config.CORS_ORIGINS == []
    client = make_app().test_client()
    for origin in ("http://localhost:3000", "https://evil.example"):
        assert "Access-Control-Allow-Origin" not in preflight(client, origin).headers


# CORS -------------------------------------------------------------------------

def test_cors_answers_only_configured_origins():
    client = make_app().test_client()
    for origin in ("http://localhost:3000", "http://127.0.0.1:5173", "https://localhost", "http://[::1]:8080"):
        allowed = preflight(client, origin)
        assert allowed.headers.getlist("Access-Control-Allow-Origin") == [origin]
        assert "Access-Control-Allow-Credentials" not in allowed.headers
    for origin in ("https://evil.example", "http://localhost.evil.example", "http://localhost:3000.evil.example",
                   "http://127.0.0.1.evil.example", "null"):
        assert "Access-Control-Allow-Origin" not in preflight(client, origin).headers
        actual = client.get("/api/v3/health", headers={"Origin": origin})
        assert actual.status_code == 200 and "Access-Control-Allow-Origin" not in actual.headers
    assert "Access-Control-Allow-Origin" not in client.get("/api/v3/health").headers


def test_cors_configured_origins_must_match_completely(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", r"https://app.example, ^https://[a-z]+\.example\.org")
    client = make_app().test_client()
    for origin in ("https://app.example", "https://docs.example.org"):
        assert preflight(client, origin).headers.getlist("Access-Control-Allow-Origin") == [origin]
    for origin in ("https://app.example.evil.net", "https://docs.example.org.evil.net",
                   "https://evil.example", "http://localhost:3000"):
        assert "Access-Control-Allow-Origin" not in preflight(client, origin).headers
    client = make_app({"CORS_ORIGINS": "https://app.example"}).test_client()  # a string, not a list
    assert preflight(client, "https://app.example").headers.getlist("Access-Control-Allow-Origin") == ["https://app.example"]
    assert "Access-Control-Allow-Origin" not in preflight(client, "https://evil.example").headers


def test_cors_wildcard_is_sent_literally_and_never_with_credentials(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "*")
    client = make_app({"CORS_SUPPORTS_CREDENTIALS": True}).test_client()
    response = preflight(client, "https://evil.example")
    assert response.headers.getlist("Access-Control-Allow-Origin") == ["*"]
    assert "Access-Control-Allow-Credentials" not in response.headers


# Health -----------------------------------------------------------------------

def test_detailed_health_reports_generic_messages_and_logs_details(monkeypatch, caplog):
    import database

    detail = "could not connect to 10.0.0.5:5432 password=hunter2"

    class BrokenUnitOfWork:
        def __enter__(self):
            raise RuntimeError(detail)

        def __exit__(self, *exc):
            return False

    def failing_check():
        raise ValueError("cannot read /srv/cevahir/secret.key")

    monkeypatch.setattr(database, "UnitOfWork", BrokenUnitOfWork)
    app = make_app()
    app.extensions["cevahir_health"].register_check("custom", failing_check)
    with caplog.at_level(logging.ERROR, logger="api.monitoring.health"):
        response = app.test_client().get("/api/v3/health/detailed")
    body = response.get_data(as_text=True)
    checks = response.json["data"]["checks"]
    assert response.status_code == 200 and response.json["data"]["status"] == "unhealthy"
    assert checks["database"] == {"status": "error", "message": "Database unavailable"}
    assert checks["custom"] == {"status": "error", "message": "Health check failed (details in the server log)"}
    for leaked in ("hunter2", "10.0.0.5", "RuntimeError", "/srv/cevahir", "secret.key"):
        assert leaked not in body
    assert detail in caplog.text and "/srv/cevahir/secret.key" in caplog.text


def test_detailed_health_reports_the_requesting_apps_own_components(monkeypatch):
    import database

    class WorkingUnitOfWork:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(database, "UnitOfWork", WorkingUnitOfWork)
    first, second = make_app(), make_app()
    first.cevahir = None
    checks = lambda app: app.test_client().get("/api/v3/health/detailed").json["data"]["checks"]
    assert checks(first)["cevahir"] == {"status": "error", "message": "Cevahir not initialized"}
    assert checks(second)["cevahir"]["status"] == "ok"


# Development server -----------------------------------------------------------

def test_development_server_defaults_to_loopback_without_debug():
    options = app_factory.parse_run_options([], environ={})
    assert (options.host, options.port, options.debug) == ("127.0.0.1", 5000, False)


def test_development_server_host_port_and_debug_are_explicit_settings():
    environ = {"CEVAHIR_API_HOST": "0.0.0.0", "PORT": "8080", "CEVAHIR_API_DEBUG": "1"}
    options = app_factory.parse_run_options([], environ=environ)
    assert (options.host, options.port, options.debug) == ("0.0.0.0", 8080, True)
    options = app_factory.parse_run_options(["--host", "::1", "--port", "9000", "--no-debug"], environ=environ)
    assert (options.host, options.port, options.debug) == ("::1", 9000, False)
    with pytest.raises(SystemExit):
        app_factory.parse_run_options([], environ={"CEVAHIR_API_DEBUG": "maybe"})


def test_main_serves_on_loopback_without_debug_by_default(monkeypatch):
    calls = []

    class RecordingApp:
        config = {}

        def run(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(app_factory, "setup_logging", lambda: None)
    monkeypatch.setattr(app_factory, "create_app", lambda: RecordingApp())
    app_factory.main([])
    assert calls == [{"host": "127.0.0.1", "port": 5000, "debug": False}]


def test_main_refuses_debug_with_the_production_configuration(monkeypatch):
    production_environment(monkeypatch)
    monkeypatch.setattr(app_factory, "setup_logging", lambda: None)
    monkeypatch.setattr(app_factory, "create_app", lambda: pytest.fail("must refuse before building the app"))
    with pytest.raises(SystemExit, match="Refusing --debug"):
        app_factory.main(["--debug"])
