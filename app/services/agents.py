from __future__ import annotations

import json
import os
import sqlite3
from copy import deepcopy
from datetime import datetime
from dotenv import load_dotenv
from typing import Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import interrupt, Command

from app.services.state import (
    TravelPlanState,
    HITLFeedback,
    ResearchOutput,
    ItineraryOutput,
    IntentValidation,
)
from app.services.tools import RESEARCH_TOOLS, PLANNER_TOOLS, TravelContext
from app.services.prompts import (
    RESEARCH_AGENT_PROMPT,
    PLANNER_AGENT_PROMPT,
    REVISION_MODIFY_INSTRUCTIONS,
    REVISION_REJECT_INSTRUCTIONS,
)

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "travel_planner.db")
if DATABASE_URL.startswith("sqlite"):
    db_path = DATABASE_URL.split("///")[-1].lstrip("./")
else:
    db_path = DATABASE_URL

_conn = sqlite3.connect(db_path, check_same_thread=False)
checkpointer = SqliteSaver(_conn)
long_term_store = InMemoryStore()

MAX_REVISIONS = 3

def _llm_reasoning(temperature: float = 1.0) -> ChatNVIDIA:
    from app.services.model_registry import get_reasoning_model
    return get_reasoning_model(with_fallback=False, temperature=temperature)

def _llm_structured(temperature: float = 0.2) -> ChatNVIDIA:
    from app.services.model_registry import get_structured_model
    return get_structured_model(temperature=temperature)

research_agent = create_agent(
    model=_llm_reasoning(),
    tools=RESEARCH_TOOLS,
    store=long_term_store,
    context_schema=TravelContext,
    response_format=ToolStrategy(ResearchOutput),  # Force tool-calling strategy
    system_prompt=RESEARCH_AGENT_PROMPT,
)

planner_agent = create_agent(
    model=_llm_reasoning(),
    tools=PLANNER_TOOLS,
    store=long_term_store,
    context_schema=TravelContext,
    response_format=ToolStrategy(ItineraryOutput),  # Force tool-calling strategy
    system_prompt=PLANNER_AGENT_PROMPT,
)

intent_validator = _llm_structured()

def validate_node(state: TravelPlanState) -> dict:
    from app.utils.logger import get_logger
    logger = get_logger("workflow.validate")
    logger.info(f"[VALIDATE] Starting validation for session {state.get('session_id')}")

    req = state.get("request")
    if not req:
        logger.error("[VALIDATE] No request provided")
        return {"status": "error", "error_message": "No travel request provided."}

    missing = [f for f in ("destination","start_date","end_date","budget_usd","travelers") if not req.get(f)]
    if missing:
        return {"status": "error", "error_message": f"Missing fields: {', '.join(missing)}"}

    try:
        start = datetime.fromisoformat(req["start_date"])
        end   = datetime.fromisoformat(req["end_date"])
        if end <= start:
            raise ValueError("end_date must be after start_date")
    except ValueError as exc:
        return {"status": "error", "error_message": str(exc)}

    if req["budget_usd"] <= 0:
        logger.error("[VALIDATE] Budget validation failed")
        return {"status": "error", "error_message": "budget_usd must be > 0"}

    logger.info("[VALIDATE] Validation passed, proceeding to intent validation")
    # Proceed to intent validation (new node)
    return {"status": "validating_intent"}

#

def run_research_node(state: TravelPlanState) -> dict:
    from app.utils.logger import get_logger, log_error
    from app.utils.callbacks import TimingCostCallback
    from app.services.model_registry import ModelRegistry
    import time

    logger = get_logger("workflow.research")
    session = state["session_id"]
    logger.info(f"[RESEARCH] Starting research for {session[:8]}")

    req      = state["request"]
    user_id  = req.get("user_id", "anonymous")

    logger.info(f"[RESEARCH] Invoking research agent for {req.get('destination')}")

    # Multi-model retry with fallback
    registry = ModelRegistry()
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Get current model (fallback on retry)
            if attempt > 0:
                current_model, model_key = registry.get_fallback()
                logger.warning(f"[RESEARCH] Retry {attempt}/{max_retries} with {model_key}")
            else:
                current_model = registry.get_model()
                model_key = registry._current_key

            # Rebuild agent with current model
            from langchain.agents import create_agent
            from langchain.agents.structured_output import ToolStrategy
            temp_research_agent = create_agent(
                model=current_model,
                tools=RESEARCH_TOOLS,
                store=long_term_store,
                context_schema=TravelContext,
                response_format=ToolStrategy(ResearchOutput),
                system_prompt=RESEARCH_AGENT_PROMPT,
            )

            # Track timing/cost
            callback = TimingCostCallback(session)
            start_time = time.perf_counter()

            # Invoke
            result = temp_research_agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=(
                            f"Research: {req['destination']}\n"
                            f"Dates: {req['start_date']} → {req['end_date']}\n"
                            f"Travellers: {req['travelers']}\n"
                            f"Interests: {', '.join(req.get('interests', []))}"
                        ))
                    ]
                },
                context=TravelContext(session_id=session, user_id=user_id),
                config={"callbacks": [callback]},
            )

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(f"[RESEARCH] Completed in {elapsed_ms}ms using {model_key}")

            # Log tool calls
            for msg in result.get("messages", []):
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        logger.info(f"[RESEARCH_TOOL] {tc.get('name')} - Args: {tc.get('args')}")
                elif hasattr(msg, "type") and msg.type == "tool":
                    logger.info(f"[RESEARCH_TOOL_RESULT] {getattr(msg, 'name', 'unknown')}: {str(msg.content)}")

            # Success
            structured: ResearchOutput = result.get("structured_response")
            research_dict = structured.model_dump() if structured else {}

            return {
                "status": "planning",
                "research_output": research_dict,
                "messages": [AIMessage(content=f"Research complete for {req['destination']}.")],
            }

        except Exception as exc:
            error_str = str(exc)
            is_retryable = (
                "500" in error_str
                or "400" in error_str  # Structured output validation failures
                or "InternalServerError" in error_str
                or "BadRequestError" in error_str
                or "Invalid JSON" in error_str  # Model generating malformed JSON
                or "EOF while parsing" in error_str
                or "timeout" in error_str.lower()
                or "rate" in error_str.lower()
            )

            if is_retryable and attempt < max_retries - 1:
                log_error(session, exc, context=f"run_research_node attempt {attempt+1}")
                logger.warning(f"[RESEARCH] Retryable error (switching model): {error_str[:150]}")
                continue  # Try next model
            else:
                # Non-retryable or final attempt
                log_error(session, exc, context="run_research_node final")
                logger.error(f"[RESEARCH] Failed after {attempt+1} attempts with all models")
                return {
                    "status": "error",
                    "error_message": f"Research failed: {type(exc).__name__}: {str(exc)[:200]}",
                }

    # Should never reach here
    return {
        "status": "error",
        "error_message": "Research failed after all retries",
    }

def run_planner_node(state: TravelPlanState) -> dict:
    from app.utils.logger import get_logger
    from app.utils.callbacks import TimingCostCallback
    import time

    logger = get_logger("workflow.planner")
    session = state["session_id"]
    logger.info(f"[PLANNER] Starting planning for {session[:8]}")

    req      = state["request"]
    user_id  = req.get("user_id", "anonymous")
    research = state.get("research_output", {})
    feedback = state.get("hitl_feedback")
    prev_draft = state.get("draft_itinerary", {})

    revision_ctx = ""
    if feedback and feedback.get("action") in ("reject", "modify"):
        revision_ctx = f"\n\n{'='*60}\nREVISION #{state.get('revision_count', 1)}\n{'='*60}\n"
        revision_ctx += f"USER FEEDBACK: {feedback.get('comments', 'No additional comments.')}\n\n"

        if feedback.get("action") == "modify":
            revision_ctx += "PREVIOUS DRAFT (preserve unmodified sections):\n"
            revision_ctx += f"{json.dumps(prev_draft, indent=2)}\n\n"

            if feedback.get("modified_itinerary"):
                revision_ctx += "USER'S REQUESTED MODIFICATIONS (apply these changes):\n"
                revision_ctx += f"{json.dumps(feedback['modified_itinerary'], indent=2)}\n\n"
                revision_ctx += "INSTRUCTIONS: Update itinerary with user's modifications while preserving all other sections unchanged.\n"
        else:
            # action == "reject" — full re-plan from scratch
            revision_ctx += "INSTRUCTIONS: User rejected previous draft. Create entirely new plan addressing their concerns.\n"

    start  = datetime.fromisoformat(req["start_date"])
    end    = datetime.fromisoformat(req["end_date"])
    n_days = max((end - start).days, 1)

    # Multi-model retry with fallback
    from app.services.model_registry import ModelRegistry
    registry = ModelRegistry()
    max_retries = 3

    for attempt in range(max_retries):
        try:
            # Get current model (fallback on retry)
            if attempt > 0:
                current_model, model_key = registry.get_fallback()
                logger.warning(f"[PLANNER] Retry {attempt}/{max_retries} with {model_key}")
            else:
                current_model = registry.get_model()
                model_key = registry._current_key

            # Rebuild agent with current model
            from langchain.agents import create_agent
            from langchain.agents.structured_output import ToolStrategy
            temp_planner_agent = create_agent(
                model=current_model,
                tools=PLANNER_TOOLS,
                store=long_term_store,
                context_schema=TravelContext,
                response_format=ToolStrategy(ItineraryOutput),
                system_prompt=PLANNER_AGENT_PROMPT,
            )

            # Track timing/cost
            callback = TimingCostCallback(session)
            start_time = time.perf_counter()

            result = temp_planner_agent.invoke(
                {
                    "messages": [
                        HumanMessage(content=(
                            f"Destination: {req['destination']}\n"
                            f"Dates: {req['start_date']} → {req['end_date']} ({n_days} days)\n"
                            f"Travellers: {req['travelers']}, Budget: ${req['budget_usd']} USD\n"
                            f"Interests: {', '.join(req.get('interests', []))}\n\n"
                            f"Research:\n{json.dumps(research, indent=2)}"
                            f"{revision_ctx}"
                        ))
                    ]
                },
                context=TravelContext(session_id=session, user_id=user_id),
                config={"callbacks": [callback]},
            )

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            logger.info(f"[PLANNER] Completed in {elapsed_ms}ms using {model_key}")

            structured: ItineraryOutput = result.get("structured_response")
            draft = structured.model_dump() if structured else {}

            return {
                "status": "awaiting_review",
                "draft_itinerary": draft,
                "hitl_feedback": None,
                "messages": [AIMessage(content="Draft itinerary ready for your review.")],
            }

        except Exception as exc:
            error_str = str(exc)
            is_retryable = (
                "500" in error_str
                or "400" in error_str  # Structured output validation failures
                or "InternalServerError" in error_str
                or "BadRequestError" in error_str
                or "Invalid JSON" in error_str  # Model generating malformed JSON
                or "EOF while parsing" in error_str
                or "timeout" in error_str.lower()
                or "rate" in error_str.lower()
            )

            if is_retryable and attempt < max_retries - 1:
                from app.utils.logger import log_error
                log_error(session, exc, context=f"run_planner_node attempt {attempt+1}")
                logger.warning(f"[PLANNER] Retryable error (switching model): {error_str[:150]}")
                continue  # Try next model
            else:
                from app.utils.logger import log_error
                log_error(session, exc, context="run_planner_node final")
                logger.error(f"[PLANNER] Failed after {attempt+1} attempts with all models")
                return {
                    "status": "error",
                    "error_message": f"Planning failed: {type(exc).__name__}: {str(exc)[:200]}",
                }

    return {
        "status": "error",
        "error_message": "Planning failed after all retries",
    }

#

def hitl_node(state: TravelPlanState) -> dict:
    from app.utils.logger import get_logger
    logger = get_logger("workflow.hitl")
    logger.info(f"[HITL] Pausing for review - {state.get('session_id')}")
    feedback: dict = interrupt({
        "message":           "Review the draft itinerary and submit your decision.",
        "draft_itinerary":   state.get("draft_itinerary", {}),
        "actions_available": ["approve", "reject", "modify"],
        "schema": {
            "action":             "approve | reject | modify",
            "comments":           "optional string",
            "modified_itinerary": "optional dict with sections to override (modify only)",
        },
    })

    return {
        "hitl_feedback": HITLFeedback(
            action=feedback.get("action", "approve"),
            comments=feedback.get("comments"),
            modified_itinerary=feedback.get("modified_itinerary"),
        )
    }

def revision_router(state: TravelPlanState) -> str:
    feedback       = state.get("hitl_feedback") or {}
    action         = feedback.get("action", "approve")
    revision_count = state.get("revision_count", 0)

    if action == "approve":
        return "finalize"
    if revision_count >= MAX_REVISIONS:
        return "finalize"          # safety net — force completion
    if action == "modify":
        return "planner"           # re-plan only
    if action == "reject":
        return "research"          # full redo
    return "finalize"

def _deep_merge(base: dict, override: dict) -> dict:
    result = deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        elif key in result and isinstance(result[key], list) and isinstance(val, list):
            # Array merge: for daily_plan modifications
            if val and isinstance(val[0], dict) and "day" in val[0]:
                result[key] = _merge_daily_plan(result[key], val)
            else:
                result[key] = val  # full replace if not daily_plan structure
        else:
            result[key] = val
    return result

def _merge_daily_plan(base_days: list[dict], override_days: list[dict]) -> list[dict]:
    result = deepcopy(base_days)
    override_map = {d["day"]: d for d in override_days if "day" in d}

    for i, day_plan in enumerate(result):
        day_num = day_plan.get("day")
        if day_num in override_map:
            result[i] = _deep_merge(day_plan, override_map[day_num])

    return result

def finalize_node(state: TravelPlanState) -> dict:
    draft    = state.get("draft_itinerary", {})
    feedback = state.get("hitl_feedback") or {}
    req      = state["request"]

    if feedback.get("action") == "modify" and feedback.get("modified_itinerary"):
        draft = _deep_merge(draft, feedback["modified_itinerary"])

    final_plan = {
        "session_id":     state["session_id"],
        "destination":    req["destination"],
        "travel_dates":   f"{req['start_date']} → {req['end_date']}",
        "travelers":      req["travelers"],
        "budget_usd":     req["budget_usd"],
        "itinerary":      draft,
        "research_notes": state.get("research_output", {}),
        "approved_at":    datetime.utcnow().isoformat() + "Z",
        "revision_count": state.get("revision_count", 0),
        "user_comments":  feedback.get("comments"),  # preserve user feedback in final plan
    }
    return {
        "status":     "complete",
        "final_plan": final_plan,
        "messages":   [AIMessage(content="Your travel plan is finalised! 🎉")],
    }

def error_node(state: TravelPlanState) -> dict:
    return {
        "status":   "error",
        "messages": [AIMessage(content=f"Error: {state.get('error_message', 'Unknown error.')}")],
    }

def increment_revision(state: TravelPlanState) -> dict:
    return {"revision_count": state.get("revision_count", 0) + 1}
