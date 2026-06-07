
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Log directory
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Log format with file/function/line for debugging
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(filename)s:%(lineno)d:%(funcName)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Console handler (stdout) with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    return logger

def get_logger(component: str = "app", enable_file: bool = True) -> logging.Logger:
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"app_{today}.log"

    return setup_logger(component, str(log_file), logging.INFO)

def get_session_logger(session_id: str) -> logging.Logger:
    safe_id = session_id.replace("-", "_")[:16]
    log_file = LOG_DIR / f"session_{safe_id}.log"

    return setup_logger(f"session.{safe_id}", str(log_file), logging.DEBUG)

def get_error_logger() -> logging.Logger:
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"errors_{today}.log"

    return setup_logger("errors", str(log_file), logging.ERROR)

# Convenience functions for common log events

def log_agent_start(session_id: str, agent_name: str, **kwargs):
    session_logger = get_session_logger(session_id)
    app_logger = get_logger("agent")

    message = f"🤖 {agent_name.upper()} AGENT STARTED"
    if kwargs:
        message += f" | {kwargs}"

    session_logger.info(message)
    app_logger.info(f"[{session_id[:8]}] {message}")

def log_agent_end(session_id: str, agent_name: str, success: bool = True, **kwargs):
    session_logger = get_session_logger(session_id)
    app_logger = get_logger("agent")

    status = "✅ COMPLETED" if success else "❌ FAILED"
    message = f"{status} | {agent_name.upper()} AGENT"
    if kwargs:
        message += f" | {kwargs}"

    session_logger.info(message)
    app_logger.info(f"[{session_id[:8]}] {message}")

def log_tool_call(
    session_id: str,
    tool_name: str,
    args: Dict[str, Any],
    agent: Optional[str] = None
):
    session_logger = get_session_logger(session_id)
    app_logger = get_logger("tool")

    agent_prefix = f"[{agent}] " if agent else ""
    message = f"🔧 {agent_prefix}TOOL CALL: {tool_name} | args={args}"

    session_logger.debug(message)
    app_logger.debug(f"[{session_id[:8]}] {message}")

def log_tool_result(
    session_id: str,
    tool_name: str,
    result: Any,
    success: bool = True
):
    session_logger = get_session_logger(session_id)

    status = "✓" if success else "✗"
    result_preview = str(result) + ("..." if len(str(result)) > 100 else "")
    message = f"{status} RESULT: {tool_name} | {result_preview}"

    session_logger.debug(message)

def log_hitl_pause(session_id: str, action: str, **kwargs):
    session_logger = get_session_logger(session_id)
    app_logger = get_logger("hitl")

    message = f"⏸️  HITL PAUSE: {action.upper()}"
    if kwargs:
        message += f" | {kwargs}"

    session_logger.info(message)
    app_logger.info(f"[{session_id[:8]}] {message}")

def log_error(
    session_id: Optional[str],
    error: Exception,
    context: Optional[str] = None
):
    import traceback

    error_logger = get_error_logger()
    app_logger = get_logger("error")

    session_prefix = f"[{session_id[:8]}] " if session_id else ""
    context_str = f" | context={context}" if context else ""

    # Extract traceback info
    tb = traceback.extract_tb(error.__traceback__)
    if tb:
        last_frame = tb[-1]
        location = f"{last_frame.filename}:{last_frame.lineno}:{last_frame.name}"
    else:
        location = "unknown"

    message = f"{session_prefix}ERROR: {type(error).__name__}: {str(error)} | location={location}{context_str}"

    error_logger.error(message, exc_info=True)
    app_logger.error(message)

    if session_id:
        session_logger = get_session_logger(session_id)
        session_logger.error(message, exc_info=True)

def log_api_request(endpoint: str, method: str, **kwargs):
    api_logger = get_logger("api")
    message = f"📥 {method} {endpoint}"
    if kwargs:
        message += f" | {kwargs}"
    api_logger.info(message)

def log_api_response(endpoint: str, status_code: int, duration_ms: float):
    api_logger = get_logger("api")
    emoji = "✅" if status_code < 400 else "❌"
    message = f"📤 {emoji} {endpoint} | {status_code} | {duration_ms:.2f}ms"
    api_logger.info(message)

def cleanup_old_logs(max_age_days: int = 7):
    import time

    cutoff_time = time.time() - (max_age_days * 24 * 3600)
    deleted_count = 0

    for log_file in LOG_DIR.glob("*.log"):
        if log_file.stat().st_mtime < cutoff_time:
            log_file.unlink()
            deleted_count += 1

    if deleted_count > 0:
        logger = get_logger("cleanup")
        logger.info(f"🗑️  Cleaned up {deleted_count} old log files")

    return deleted_count
