"""
main.py — FastAPI application entry point.

Run with:
    uvicorn main:app --reload --port 8000

Production:
    gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.api.v1.routes import router

# Load environment variables from .env file
load_dotenv()

def validate_environment() -> dict[str, str]:
    """
    Validate required API keys and environment variables on startup.

    Returns dict of warnings if keys missing (non-blocking).
    Logs errors but allows startup for development with mock fallbacks.
    """
    warnings = {}

    # Required API keys
    if not os.getenv("NVIDIA_API_KEY"):
        warnings["NVIDIA_API_KEY"] = "Missing — LLM calls will fail. Get free key at https://build.nvidia.com/explore/discover"

    if not os.getenv("SERPER_API_KEY"):
        warnings["SERPER_API_KEY"] = "Missing — web search will return empty results. Get free key at https://serper.dev"

    # Optional API keys (have fallbacks)
    if not os.getenv("EXCHANGERATE_API_KEY"):
        warnings["EXCHANGERATE_API_KEY"] = "Missing — using mock exchange rates (OK for dev)"

    # Model configuration
    model = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct")
    valid_models = [
        "meta/llama-3.1-405b-instruct",
        "meta/llama-3.1-70b-instruct",
        "meta/llama-3.1-8b-instruct",
        "mistralai/mixtral-8x7b-instruct-v0.1",
    ]
    if model not in valid_models:
        warnings["NVIDIA_MODEL"] = f"Unknown model '{model}' — may cause errors. Valid: {', '.join(valid_models)}"

    return warnings

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle with environment validation."""
    print("\n" + "="*60)
    print("[START] AI Travel Planner API - Starting Up")
    print("="*60)

    # Validate environment
    warnings = validate_environment()
    if warnings:
        print("\n[WARNING] ENVIRONMENT WARNINGS:")
        for key, msg in warnings.items():
            print(f"   * {key}: {msg}")
        print("\n[INFO] Copy .env.example to .env and fill in your API keys.\n")
    else:
        print("[OK] All environment variables configured correctly.\n")

    # Log configuration
    print(f"[LLM] Model: {os.getenv('NVIDIA_MODEL', 'meta/llama-3.1-70b-instruct')} (NVIDIA)")
    print(f"[DB] Database: travel_planner.db (SQLite + InMemoryStore)")
    print(f"[HITL] Checkpointer: SqliteSaver (enables pause/resume)")
    print(f"[HTTP] Listening on: http://localhost:{os.getenv('PORT', '8000')}")
    print("="*60 + "\n")

    yield

    print("\n[STOP] Travel Planner API shutting down\n")

app = FastAPI(
    title="AI Travel Planner",
    description=(
        "Multi-agent travel planning system with human-in-the-loop approval.\n\n"
        "**Architecture:** Research Agent → Planner Agent → HITL Review → Finalize\n\n"
        "**Tools:** Serper (web search), ExchangeRate API (currency), budget allocator, packing list generator\n\n"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include plan routes
app.include_router(router)

# Root + Health endpoints

@app.get("/", tags=["Meta"], include_in_schema=False)
async def root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")

@app.get("/health", tags=["Meta"])
async def health() -> dict:
    """
    Health check endpoint.

    Returns API status + environment validation.
    """
    warnings = validate_environment()

    return {
        "status": "healthy" if not warnings else "degraded",
        "version": "1.0.0",
        "model": os.getenv("LLM_MODEL"),
        "warnings": warnings if warnings else None,
    }

@app.get("/status", tags=["Meta"])
async def status() -> dict:
    """
    Detailed status endpoint showing configuration.
    """
    return {
        "service": "AI Travel Planner",
        "version": "1.0.0",
        "environment": {
            "llm_provider": "NVIDIA AI Endpoints",
            "llm_model": os.getenv("NVIDIA_MODEL", "meta/llama-3.1-70b-instruct"),
            "nvidia_api_configured": bool(os.getenv("NVIDIA_API_KEY")),
            "serper_configured": bool(os.getenv("SERPER_API_KEY")),
            "exchangerate_configured": bool(os.getenv("EXCHANGERATE_API_KEY")),
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "create_plan": "POST /plan",
            "get_plan": "GET /plan/{session_id}",
            "review_plan": "POST /plan/{session_id}/review",
            "final_plan": "GET /plan/{session_id}/final",
            "stream": "GET /plan/{session_id}/stream",
        }
    }

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,
        log_level="info",
    )
