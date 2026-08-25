import time

import jwt
import pytest

from auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)


def test_hash_password_produces_salt_and_digest():
    hashed = hash_password("correct-horse-battery-staple")
    assert "$" in hashed
    salt, digest = hashed.split("$")
    assert len(salt) == 32  # 16 bytes hex-encoded
    assert len(digest) > 0


def test_hash_password_is_non_deterministic():
    """Same password, different salt each time — two hashes of the same
    password must never be equal, or a database leak would let an
    attacker spot duplicate passwords across accounts."""
    h1 = hash_password("same-password")
    h2 = hash_password("same-password")
    assert h1 != h2


def test_verify_password_accepts_correct_password():
    hashed = hash_password("my-real-password")
    assert verify_password("my-real-password", hashed) is True


def test_verify_password_rejects_wrong_password():
    hashed = hash_password("my-real-password")
    assert verify_password("wrong-password", hashed) is False


def test_verify_password_rejects_malformed_hash():
    """A stored hash without the salt$digest format (corrupted data,
    or an old format from a future migration) must fail closed, not throw."""
    assert verify_password("anything", "not-a-valid-hash-format") is False


def test_create_and_decode_access_token_roundtrip():
    token = create_access_token(subject="alice", role="claims_adjuster")
    payload = decode_access_token(token)
    assert payload["sub"] == "alice"
    assert payload["role"] == "claims_adjuster"


def test_decode_rejects_tampered_token():
    token = create_access_token(subject="alice", role="claims_adjuster")
    tampered = token[:-4] + "abcd"  # corrupt the signature
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(tampered)


def test_decode_rejects_token_signed_with_different_secret():
    """A token forged with a guessed/different secret must never verify —
    this is the actual security property JWT signing exists to provide."""
    forged = jwt.encode(
        {"sub": "alice", "role": "admin"}, "wrong-secret-attacker-guessed", algorithm="HS256"
    )
    with pytest.raises(jwt.PyJWTError):
        decode_access_token(forged)


def test_token_includes_expiry_in_the_future():
    token = create_access_token(subject="alice", role="claims_adjuster")
    payload = decode_access_token(token)
    assert payload["exp"] > time.time()


def test_expired_token_raises_expired_signature_error():
    import auth.security as security_module

    original_expiry = security_module.ACCESS_TOKEN_EXPIRE_MINUTES
    try:
        security_module.ACCESS_TOKEN_EXPIRE_MINUTES = -1  # already-expired token
        token = create_access_token(subject="alice", role="claims_adjuster")
    finally:
        security_module.ACCESS_TOKEN_EXPIRE_MINUTES = original_expiry

    with pytest.raises(jwt.ExpiredSignatureError):
        decode_access_token(token)
