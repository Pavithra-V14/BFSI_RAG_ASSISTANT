from __future__ import annotations

import hashlib
import os
import secrets
import time
from datetime import datetime, timedelta, timezone

import jwt

SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-insecure-secret-change-me")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))
PBKDF2_ITERATIONS = 260_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ITERATIONS)
    return f"{salt}${digest.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        salt, digest_hex = stored_hash.split("$")
    except ValueError:
        return False
    candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), PBKDF2_ITERATIONS)
    return secrets.compare_digest(candidate.hex(), digest_hex)


def create_access_token(subject: str, role: str, extra: dict | None = None) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": subject,
        "role": role,
        "iat": now,
        "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        "jti": secrets.token_hex(16), 
        **(extra or {}),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict:
    """Raises jwt.PyJWTError on invalid/expired token — caller converts to 401."""
    return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])