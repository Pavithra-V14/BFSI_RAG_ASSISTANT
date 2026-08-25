import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture()
def isolated_auth_store(monkeypatch):
    """Each test gets a throwaway SQLite file, not the real data/app.db —
    auth/store.py resolves DB_PATH at import time, so we patch the module
    attribute directly rather than an env var."""
    import auth.store as store

    with tempfile.TemporaryDirectory() as tmpdir:
        monkeypatch.setattr(store, "DB_PATH", Path(tmpdir) / "test_app.db")
        store.init_db()
        yield store


def test_valid_roles_excludes_relationship_manager(isolated_auth_store):
    """Regression guard for the 2026-08 role consolidation — this role
    was deliberately dropped, must never silently reappear."""
    assert "relationship_manager" not in isolated_auth_store.VALID_ROLES


def test_self_serve_roles_is_only_the_lowest_privilege_role(isolated_auth_store):
    assert isolated_auth_store.SELF_SERVE_ROLES == {"claims_adjuster"}
    assert "admin" not in isolated_auth_store.SELF_SERVE_ROLES
    assert "compliance_officer" not in isolated_auth_store.SELF_SERVE_ROLES


def test_create_user_rejects_invalid_role(isolated_auth_store):
    with pytest.raises(ValueError):
        isolated_auth_store.create_user("bob", "bob@test.com", "hashed", "not_a_real_role")


def test_create_user_succeeds_with_valid_role(isolated_auth_store):
    user = isolated_auth_store.create_user("carol", "carol@test.com", "hashed", "claims_adjuster")
    assert user.username == "carol"
    assert user.role == "claims_adjuster"
    assert user.id is not None


def test_username_or_email_exists_detects_duplicate_username(isolated_auth_store):
    isolated_auth_store.create_user("dave", "dave@test.com", "hashed", "claims_adjuster")
    assert isolated_auth_store.username_or_email_exists("dave", "different@test.com") is True


def test_username_or_email_exists_detects_duplicate_email(isolated_auth_store):
    isolated_auth_store.create_user("dave", "dave@test.com", "hashed", "claims_adjuster")
    assert isolated_auth_store.username_or_email_exists("different_username", "dave@test.com") is True


def test_username_or_email_exists_false_for_new_identity(isolated_auth_store):
    isolated_auth_store.create_user("dave", "dave@test.com", "hashed", "claims_adjuster")
    assert isolated_auth_store.username_or_email_exists("nobody", "nobody@test.com") is False


def test_get_user_by_username_returns_none_for_unknown_user(isolated_auth_store):
    assert isolated_auth_store.get_user_by_username("ghost") is None


def test_update_role_promotes_user(isolated_auth_store):
    user = isolated_auth_store.create_user("eve", "eve@test.com", "hashed", "claims_adjuster")
    ok = isolated_auth_store.update_role(user.id, "compliance_officer")
    assert ok is True
    row = isolated_auth_store.get_user_by_id(user.id)
    assert row["role"] == "compliance_officer"


def test_update_role_rejects_invalid_role(isolated_auth_store):
    user = isolated_auth_store.create_user("frank", "frank@test.com", "hashed", "claims_adjuster")
    with pytest.raises(ValueError):
        isolated_auth_store.update_role(user.id, "super_admin")


def test_update_role_returns_false_for_nonexistent_user(isolated_auth_store):
    assert isolated_auth_store.update_role(999999, "admin") is False


def test_list_users_returns_all_created_users(isolated_auth_store):
    isolated_auth_store.create_user("g1", "g1@test.com", "hashed", "claims_adjuster")
    isolated_auth_store.create_user("g2", "g2@test.com", "hashed", "admin")
    usernames = {row["username"] for row in isolated_auth_store.list_users()}
    assert usernames == {"g1", "g2"}
