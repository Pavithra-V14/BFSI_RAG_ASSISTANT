import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def isolated_session_store(monkeypatch):
    import sessions.store as session_store

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(session_store, "DB_PATH", Path(tmpdir) / "test_app.db")
        session_store.init_db()
        yield session_store


def test_create_session_returns_session_with_default_title(isolated_session_store):
    s = isolated_session_store.create_session(user_id=1)
    assert s.title == "New chat"
    assert s.user_id == 1
    assert s.session_id


def test_list_sessions_returns_only_that_users_sessions(isolated_session_store):
    isolated_session_store.create_session(user_id=1)
    isolated_session_store.create_session(user_id=1)
    isolated_session_store.create_session(user_id=2)
    assert len(isolated_session_store.list_sessions(user_id=1)) == 2
    assert len(isolated_session_store.list_sessions(user_id=2)) == 1


def test_get_session_enforces_ownership_at_the_query_level(isolated_session_store):
    """This is the actual security property sessions/store.py exists to
    guarantee — a session_id from another user must return None, not the
    other user's session, even though the row physically exists."""
    s = isolated_session_store.create_session(user_id=1)
    assert isolated_session_store.get_session(s.session_id, user_id=1) is not None
    assert isolated_session_store.get_session(s.session_id, user_id=2) is None


def test_get_session_returns_none_for_nonexistent_session(isolated_session_store):
    assert isolated_session_store.get_session("not-a-real-uuid", user_id=1) is None


def test_add_and_get_messages_roundtrip(isolated_session_store):
    s = isolated_session_store.create_session(user_id=1)
    isolated_session_store.add_message(s.session_id, "user", "What is the waiting period?")
    isolated_session_store.add_message(
        s.session_id, "assistant", "30 days.", citations=["doc::c1"], grounded=True
    )
    messages = isolated_session_store.get_messages(s.session_id, user_id=1)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is the waiting period?"
    assert messages[1]["citations"] == ["doc::c1"]
    assert messages[1]["grounded"] is True


def test_get_messages_enforces_ownership_returns_empty_not_error(isolated_session_store):
    """Deliberately returns [] rather than raising or 404ing for a
    session that exists but isn't owned by the caller — avoids confirming
    to an attacker that a guessed session_id is real (see app.py's
    /sessions/{id}/messages route, which relies on this exact behavior)."""
    s = isolated_session_store.create_session(user_id=1)
    isolated_session_store.add_message(s.session_id, "user", "secret question")
    assert isolated_session_store.get_messages(s.session_id, user_id=2) == []


def test_messages_ordered_chronologically(isolated_session_store):
    s = isolated_session_store.create_session(user_id=1)
    for i in range(5):
        isolated_session_store.add_message(s.session_id, "user", f"message {i}")
    messages = isolated_session_store.get_messages(s.session_id, user_id=1)
    contents = [m["content"] for m in messages]
    assert contents == [f"message {i}" for i in range(5)]


def test_rename_session_if_first_message_only_renames_once(isolated_session_store):
    s = isolated_session_store.create_session(user_id=1)
    isolated_session_store.rename_session_if_first_message(s.session_id, 1, "What is the waiting period?")
    renamed = isolated_session_store.get_session(s.session_id, user_id=1)
    assert renamed["title"] == "What is the waiting period?"

    # a second call must NOT overwrite the title with a different query
    isolated_session_store.rename_session_if_first_message(s.session_id, 1, "a totally different question")
    still = isolated_session_store.get_session(s.session_id, user_id=1)
    assert still["title"] == "What is the waiting period?"


def test_rename_truncates_long_titles(isolated_session_store):
    s = isolated_session_store.create_session(user_id=1)
    long_query = "a" * 100
    isolated_session_store.rename_session_if_first_message(s.session_id, 1, long_query)
    renamed = isolated_session_store.get_session(s.session_id, user_id=1)
    assert len(renamed["title"]) <= 51  # 50 chars + ellipsis


def test_delete_session_removes_it_and_its_messages(isolated_session_store):
    s = isolated_session_store.create_session(user_id=1)
    isolated_session_store.add_message(s.session_id, "user", "hello")
    ok = isolated_session_store.delete_session(s.session_id, user_id=1)
    assert ok is True
    assert isolated_session_store.get_session(s.session_id, user_id=1) is None


def test_delete_session_enforces_ownership(isolated_session_store):
    """A different user's session_id must not be deletable, even by guessing
    a valid UUID — delete_session() must return False, not silently succeed
    on someone else's data."""
    s = isolated_session_store.create_session(user_id=1)
    ok = isolated_session_store.delete_session(s.session_id, user_id=2)
    assert ok is False
    # session must still exist for its real owner
    assert isolated_session_store.get_session(s.session_id, user_id=1) is not None


def test_delete_nonexistent_session_returns_false(isolated_session_store):
    assert isolated_session_store.delete_session("not-a-real-uuid", user_id=1) is False
