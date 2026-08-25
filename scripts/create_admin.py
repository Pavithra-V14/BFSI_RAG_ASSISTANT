"""
One-time admin bootstrap — creates the FIRST admin account.

Why this exists as a script, not an API endpoint: /auth/register only
ever creates the lowest-privilege role (see auth/store.py's
SELF_SERVE_ROLES), and promoting a user to admin requires an existing
admin (PATCH /admin/users/{id}/role). That's correct and intentional —
but it means a fresh deployment with zero users has no way to create its
first admin through the API at all. This script is the deliberate,
narrow exception: run once, by whoever has shell access to the server
(which is already a higher trust bar than "anyone who can reach the API"),
not exposed over HTTP at all.

Usage:
    python -m scripts.create_admin --username admin1 --email admin1@company.com

Prompts for a password interactively (never accepted as a CLI arg, so it
never ends up in shell history or process listings).
"""
from __future__ import annotations

import argparse
import getpass
import sys

import auth.store as store
from auth.security import hash_password


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--username", required=True)
    ap.add_argument("--email", required=True)
    args = ap.parse_args()

    if store.username_or_email_exists(args.username, args.email):
        print(f"Error: username or email already registered.", file=sys.stderr)
        sys.exit(1)

    existing_admins = [u for u in store.list_users() if u["role"] == "admin"]
    if existing_admins:
        print(
            f"Note: {len(existing_admins)} admin account(s) already exist "
            f"({', '.join(u['username'] for u in existing_admins)}). "
            "Proceeding will create an additional admin — use PATCH "
            "/admin/users/{id}/role via the API instead if you're just "
            "promoting an existing user.",
        )
        if input("Continue creating a new admin account? [y/N] ").strip().lower() != "y":
            sys.exit(0)

    password = getpass.getpass("Password (min 8 chars): ")
    if len(password) < 8:
        print("Error: password must be at least 8 characters.", file=sys.stderr)
        sys.exit(1)
    confirm = getpass.getpass("Confirm password: ")
    if password != confirm:
        print("Error: passwords did not match.", file=sys.stderr)
        sys.exit(1)

    user = store.create_user(args.username, args.email, hash_password(password), "admin")
    print(f"Created admin account: {user.username} (id={user.id})")


if __name__ == "__main__":
    main()
