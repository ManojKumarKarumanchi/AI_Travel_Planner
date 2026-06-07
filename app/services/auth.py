"""
auth.py — JWT authentication for FastAPI endpoints.

Pattern:
  - User registers/logs in → receives JWT access token
  - Protected endpoints use Depends(get_current_user) → validates JWT
  - Token contains user_id + expiration
  - Passwords hashed with bcrypt

Usage:
  @router.post("/plan", dependencies=[Depends(get_current_user)])
  async def create_plan(body: PlanRequest, user: User = Depends(get_current_user)):
      # user.id available here
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr

# Configuration from environment

SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-in-prod")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

if SECRET_KEY == "dev-secret-change-in-prod":
    import warnings
    warnings.warn("Using default SECRET_KEY — generate a secure key for production!")

# Password hashing

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

# JWT token creation

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token.

    Args:
        data: Payload dict (typically {"sub": user_id, "email": user_email})
        expires_delta: Token lifetime (defaults to ACCESS_TOKEN_EXPIRE_MINUTES)

    Returns:
        Encoded JWT string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Pydantic models

class User(BaseModel):
    """User model (matches database schema)."""
    id: str
    email: EmailStr
    is_active: bool = True

class TokenData(BaseModel):
    """JWT token payload."""
    user_id: Optional[str] = None
    email: Optional[str] = None

class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"

# In-memory user store (replace with database in production)

# Mock users for development — in production, query from DATABASE_URL
# Lazy init to avoid bcrypt hash at import time (causes passlib test issues)
_MOCK_USERS = {}

def get_user_by_email(email: str) -> Optional[dict]:
    """Retrieve user from store (mock implementation)."""
    # Lazy init demo user
    if not _MOCK_USERS and email == "demo@example.com":
        _MOCK_USERS["demo@example.com"] = {
            "id": "user_demo_123",
            "email": "demo@example.com",
            "hashed_password": hash_password("demo123"),
            "is_active": True,
        }
    return _MOCK_USERS.get(email)

def create_user(email: str, password: str) -> dict:
    """Create new user (mock implementation)."""
    import uuid
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    user_data = {
        "id": user_id,
        "email": email,
        "hashed_password": hash_password(password),
        "is_active": True,
    }
    _MOCK_USERS[email] = user_data
    return user_data

# FastAPI dependencies

security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    FastAPI dependency: validate JWT token and return current user.

    Usage:
        @router.get("/me")
        async def read_me(user: User = Depends(get_current_user)):
            return user

    Raises:
        HTTPException 401: if token invalid/expired/missing
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        email: str = payload.get("email")

        if user_id is None or email is None:
            raise credentials_exception

        token_data = TokenData(user_id=user_id, email=email)
    except JWTError:
        raise credentials_exception

    # Fetch user from store (in production, query database)
    user_dict = get_user_by_email(token_data.email)
    if user_dict is None:
        raise credentials_exception

    if not user_dict.get("is_active", False):
        raise HTTPException(status_code=400, detail="Inactive user")

    return User(id=user_dict["id"], email=user_dict["email"], is_active=user_dict["is_active"])

# Optional: Admin-only dependency

async def get_admin_user(user: User = Depends(get_current_user)) -> User:
    """
    FastAPI dependency: validate user is admin.

    Usage:
        @router.delete("/plans/all", dependencies=[Depends(get_admin_user)])
        async def delete_all_plans():
            # Only admins reach here
    """
    # In production, check user.role == "admin" from database
    if user.email not in ["admin@example.com"]:  # mock admin check
        raise HTTPException(status_code=403, detail="Admin access required")
    return user