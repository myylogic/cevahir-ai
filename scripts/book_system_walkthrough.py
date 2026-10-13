"""Executable book cases: actual chat/context/storage with a controlled backend.

Uses a temporary SQLite database and no model/tokenizer assets. The backend is a
teaching fixture, so these cases make no neural-quality or throughput claim.
Default mode checks the recorded result; --write explicitly refreshes evidence.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEPS = ROOT / ".evolution/deps"
if DEPS.exists():
    sys.path.insert(0, str(DEPS))

import sqlalchemy
from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from chatting_management import ChattingConfig, ChattingManager
from chatting_management.exceptions import CevahirIntegrationError, SessionAccessDeniedError
from cognitive_management.cognitive_types import CognitiveInput, CognitiveOutput
from cognitive_management.v2.utils.request_scope import current_scope, scoped_request
from database import UnitOfWork
from database.connection import db
from database.models import Base, Message, User

OUTPUT = ROOT / "docs/book/evidence/system_walkthrough.json"


class ControlledBackend:
    """Returns fixed text after observing the real context and database boundary."""

    def __init__(self):
        self.calls = []
        self.manager = None
        self.fail = False
        self.memory_service = SimpleNamespace(revision=0)

    def process(self, text, state, **kwargs):
        return self.handle(state, CognitiveInput(user_message=text), **kwargs)

    @scoped_request
    def handle(self, state, request, **kwargs):
        rows = self.manager.conversation_storage.get_conversation_history(state.session_id)
        scope = json.loads(current_scope())
        assert scope == [state.metadata["user_id"], state.session_id]
        self.calls.append({
            "text": request.user_message,
            "history": [dict(item) for item in state.history],
            "persisted_count_at_generation": len(rows),
            "scope_user": scope[0],
            "scope_session_matches_state": scope[1] == state.session_id,
        })
        if self.fail:
            raise RuntimeError("injected backend failure")
        return CognitiveOutput(text="controlled answer", used_mode="think1")


def database_cases():
    with tempfile.TemporaryDirectory(prefix="cevahir-book-system-") as directory:
        engine = create_engine("sqlite:///" + str(Path(directory) / "teaching.db"))
        Base.metadata.create_all(engine)
        factory = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
        try:
            with patch.object(db, "get_session_factory", return_value=factory):
                with UnitOfWork() as uow:
                    for user in ("alice", "bob"):
                        uow.users.create(User(user_id=user, email=user + "@example.invalid"))
                backend = ControlledBackend()
                cfg = ChattingConfig(enable_user_memory=False, enable_semantic_search=False)
                manager = ChattingManager(cfg, backend)
                backend.manager = manager
                result = {}

                sid = manager.create_session("alice", title="successful call")["session_id"]
                manager.send_message("alice", sid, "first input")
                first = backend.calls[-1]
                rows = manager.get_conversation_history(sid, "alice")
                assert first["history"] == [] and first["persisted_count_at_generation"] == 0
                assert [(r["role"], r["content"]) for r in rows] == [
                    ("user", "first input"), ("assistant", "controlled answer")]
                assert current_scope() == "local"
                result["successful_call"] = {"backend_observation": first,
                    "persisted_roles_after_return": [r["role"] for r in rows],
                    "current_input_row_count": sum(r["content"] == "first input" for r in rows),
                    "scope_restored_after_call": current_scope() == "local"}

                # Repeating the same operation is deliberately a second request.
                manager.send_message("alice", sid, "first input")
                rows = manager.get_conversation_history(sid, "alice")
                assert len(rows) == 4 and len(backend.calls) == 2
                result["identical_request_repeated"] = {"backend_calls": 2,
                    "rows_after_two_calls": len(rows),
                    "user_rows_with_identical_content": sum(r["role"] == "user" and r["content"] == "first input" for r in rows),
                    "second_call_history_count": len(backend.calls[-1]["history"])}

                before = len(backend.calls)
                try:
                    manager.send_message("bob", sid, "forbidden input")
                except SessionAccessDeniedError:
                    pass
                else:
                    raise AssertionError("Wrong owner must fail before generation")
                assert len(backend.calls) == before
                result["session_ownership"] = {"error": "SessionAccessDeniedError", "additional_backend_calls": 0,
                    "rows_unchanged": len(manager.get_conversation_history(sid, "alice")) == 4}

                failed_sid = manager.create_session("alice", title="backend failure")["session_id"]
                backend.fail = True
                try:
                    manager.send_message("alice", failed_sid, "unanswered input")
                except CevahirIntegrationError:
                    pass
                else:
                    raise AssertionError("Backend failure must propagate")
                finally:
                    backend.fail = False
                assert manager.get_conversation_history(failed_sid, "alice") == []
                assert current_scope() == "local"
                result["backend_failure"] = {"error": "CevahirIntegrationError", "persisted_rows": 0,
                    "scope_restored_after_error": current_scope() == "local"}

                partial_sid = manager.create_session("alice", title="second write failure")["session_id"]
                original = manager.conversation_manager.add_message
                def fail_assistant_write(**kwargs):
                    if kwargs["role"] == "assistant":
                        raise RuntimeError("injected assistant persistence failure")
                    return original(**kwargs)
                before = len(backend.calls)
                with patch.object(manager.conversation_manager, "add_message", side_effect=fail_assistant_write):
                    try:
                        manager.send_message("alice", partial_sid, "partial input")
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("Second persistence failure must propagate")
                rows = manager.get_conversation_history(partial_sid, "alice")
                assert [r["role"] for r in rows] == ["user"]
                result["second_write_failure"] = {"backend_calls": len(backend.calls) - before,
                    "persisted_roles": [r["role"] for r in rows], "normal_return": False,
                    "note": "Injected failure between separate production storage transactions; no hardware crash simulated."}

                budget_sid = manager.create_session("alice", title="budget")["session_id"]
                with UnitOfWork() as uow:
                    for index, content in enumerate(("old old old!", "abcdefgh", "ijklmnop")):
                        uow.messages.create(Message(message_id="budget-" + str(index), session_id=budget_sid,
                            role="user", content=content, created_at=datetime(2020, 1, 1) + timedelta(seconds=index)))
                budget = manager.context_builder.build_context("alice", budget_sid, "current", max_tokens=4)
                assert [r["content"] for r in budget.history] == ["abcdefgh", "ijklmnop"]
                overlong_sid = manager.create_session("alice", title="overlong newest")["session_id"]
                with UnitOfWork() as uow:
                    for index, content in enumerate(("tiny", "x" * 20)):
                        uow.messages.create(Message(message_id="overlong-" + str(index), session_id=overlong_sid,
                            role="user", content=content, created_at=datetime(2020, 1, 1) + timedelta(seconds=index)))
                assert manager.context_builder._get_recent_history(overlong_sid, 4) == []
                result["history_budget"] = {"budget": 4, "selected_contents": [r["content"] for r in budget.history],
                    "estimated_tokens": manager.context_builder._estimate_tokens(budget.history),
                    "oversized_newest_message_selected_history": [], "current_message_not_in_history": True}

                memory_manager = ChattingManager(ChattingConfig(enable_user_memory=True, enable_semantic_search=False,
                                                max_context_tokens=1), backend)
                memory_manager.memory_storage.add_memory("alice", "preference", "m" * 40, priority="high")
                memory_state = memory_manager.context_builder.build_context("alice", failed_sid, "query")
                memory_estimate = memory_manager.context_builder._estimate_tokens(memory_state.history)
                assert len(memory_state.history) == 1 and memory_estimate > 1
                result["memory_added_after_history_budget"] = {"configured_history_budget": 1,
                    "final_history_estimate": memory_estimate, "roles": [r["role"] for r in memory_state.history],
                    "note": "The approximate recent-history budget is not a final prompt token cap."}
                return result
        finally:
            factory.remove()
            engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    logging.disable(logging.CRITICAL)
    cases = database_cases()
    if args.write:
        evidence = {"date": "2026-09-21", "kind": "executed_book_examples",
            "environment": {"python": platform.python_version(), "sqlalchemy": sqlalchemy.__version__,
                            "database": "temporary SQLite", "model": "controlled backend fixture"},
            "cases": cases, "limits": ["No language model, tokenizer, HTTP transport or hardware crash is exercised",
                "Actual ChattingManager, ContextBuilder, storage, UnitOfWork and request-scope decorator execute",
                "Failure injection documents the present transaction boundary; not a guarantee against every failure mode"]}
        OUTPUT.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    else:
        assert cases == json.loads(OUTPUT.read_text(encoding="utf-8"))["cases"]
    print(json.dumps({"status": "passed", "example_groups": len(cases), "mode": "record" if args.write else "check"}))


if __name__ == "__main__":
    main()
