"""
auth_routes.py — Registration and login endpoints.

Endpoints:
  POST /auth/register — Create new user account
  POST /auth/login    — Get JWT access token
  GET  /auth/me       — Get current user info (requires auth)
"""

from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr, Field

from app.services.auth import (
    create_access_token,
    verify_password,
    get_user_by_email,
    create_user,
    get_current_user,
    User,
    Token,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])

# Request/Response models

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6, example="securepass123")

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class UserResponse(BaseModel):
    id: str
    email: str
    message: str

# POST /auth/register

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest):
    """
    Register a new user account.

    Returns user info (without password). Use /auth/login to get JWT token.
    """
    # Check if user already exists
    existing_user = get_user_by_email(body.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create user
    user = create_user(email=body.email, password=body.password)

    return UserResponse(
        id=user["id"],
        email=user["email"],
        message="User registered successfully. Use /auth/login to get access token."
    )

# POST /auth/login

@router.post("/login", response_model=Token)
async def login(body: LoginRequest):
    """
    Login and receive JWT access token.

    Use token in Authorization header: `Authorization: Bearer <token>`
    """
    user = get_user_by_email(body.email)

    # Verify user exists and password matches
    if not user or not verify_password(body.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.get("is_active", False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account"
        )

    # Create JWT token
    access_token = create_access_token(data={"sub": user["id"], "email": user["email"]})

    return Token(access_token=access_token, token_type="bearer")

# GET /auth/me

@router.get("/me", response_model=User)
async def read_current_user(user: User = Depends(get_current_user)):
    """
    Get current authenticated user info.

    Requires valid JWT token in Authorization header.
    """
    return user
