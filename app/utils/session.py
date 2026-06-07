"""
session.py — Session management utilities.

Functions:
  - generate_session_id: Create unique session identifier
  - validate_session_id: Check if session ID is valid
  - get_session_dir: Get directory for session artifacts
  - cleanup_old_sessions: Remove old session data
"""

import re
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Session data directory
SESSION_DIR = Path(__file__).parent.parent.parent / "sessions"
SESSION_DIR.mkdir(exist_ok=True)


def generate_session_id() -> str:
    """
    Generate unique session identifier.

    Format: UUID4 (e.g., "550e8400-e29b-41d4-a716-446655440000")

    Returns:
        Unique session ID string
    """
    return str(uuid.uuid4())


def validate_session_id(session_id: str) -> bool:
    """
    Validate session ID format.

    Args:
        session_id: Session identifier to validate

    Returns:
        True if valid UUID4 format, False otherwise
    """
    uuid_pattern = r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    return bool(re.match(uuid_pattern, session_id, re.IGNORECASE))


def get_session_dir(session_id: str, create: bool = True) -> Path:
    """
    Get directory path for session artifacts.

    Directory structure:
      sessions/{session_id}/
        - graph.png
        - metadata.json
        - state.json
        - logs.txt

    Args:
        session_id: Session identifier
        create: Create directory if doesn't exist

    Returns:
        Path to session directory
    """
    safe_id = session_id.replace("-", "_")[:16]
    session_path = SESSION_DIR / safe_id

    if create:
        session_path.mkdir(parents=True, exist_ok=True)

    return session_path


def save_session_state(session_id: str, state: dict):
    """
    Save session state to JSON file.

    Args:
        session_id: Session identifier
        state: State dictionary to save
    """
    import json

    session_path = get_session_dir(session_id)
    state_file = session_path / "state.json"

    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, default=str)  # default=str handles datetime

    return str(state_file)


def load_session_state(session_id: str) -> Optional[dict]:
    """
    Load session state from JSON file.

    Args:
        session_id: Session identifier

    Returns:
        State dict or None if not found
    """
    import json

    session_path = get_session_dir(session_id, create=False)
    state_file = session_path / "state.json"

    if state_file.exists():
        with open(state_file, "r") as f:
            return json.load(f)

    return None


def cleanup_old_sessions(max_age_hours: int = 48):
    """
    Clean up session directories older than specified hours.

    Args:
        max_age_hours: Delete sessions older than this many hours

    Returns:
        Number of sessions deleted
    """
    import shutil
    import time

    cutoff_time = time.time() - (max_age_hours * 3600)
    deleted_count = 0

    for session_path in SESSION_DIR.iterdir():
        if session_path.is_dir() and session_path.stat().st_mtime < cutoff_time:
            shutil.rmtree(session_path)
            deleted_count += 1

    return deleted_count


def get_active_sessions() -> list[str]:
    """
    Get list of all active session IDs.

    Returns:
        List of session ID strings
    """
    sessions = []

    for session_path in SESSION_DIR.iterdir():
        if session_path.is_dir():
            # Try to reconstruct session ID from directory name
            sessions.append(session_path.name)

    return sessions


def session_exists(session_id: str) -> bool:
    """
    Check if session directory exists.

    Args:
        session_id: Session identifier

    Returns:
        True if session exists
    """
    session_path = get_session_dir(session_id, create=False)
    return session_path.exists()
