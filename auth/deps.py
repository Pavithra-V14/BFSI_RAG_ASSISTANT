from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from auth.security import decode_access_token
from auth.store import get_user_by_username, is_token_revoked

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

class CurrentUser:
    def __init__(self, username: str, role: str, user_id: int, jti: str | None = None):
        self.username = username
        self.role = role
        self.user_id = user_id
        self.jti = jti  

def get_current_user(token: str = Depends(oauth2_scheme)) -> CurrentUser:
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token expired")
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid token")

    jti = payload.get("jti")
    if jti and is_token_revoked(jti):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "token has been revoked")

    username = payload.get("sub")
    row = get_user_by_username(username)
    if row is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user no longer exists")

    return CurrentUser(username=row["username"], role=row["role"], user_id=row["id"], jti=jti)


def require_role(*allowed_roles: str):
    """Dependency factory — e.g. Depends(require_role('admin', 'compliance_officer'))."""
    def _check(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in allowed_roles:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"role '{user.role}' is not permitted for this action (requires one of {allowed_roles})",
            )
        return user
    return _check