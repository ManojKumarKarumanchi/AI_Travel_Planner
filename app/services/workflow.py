
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from app.services.state import TravelPlanState
from app.services.agents import (
    checkpointer,          # SqliteSaver singleton from agents.py
    long_term_store,       # InMemoryStore singleton from agents.py
    validate_node,
    run_research_node,
    run_planner_node,
    hitl_node,
    revision_router,
    finalize_node,
    error_node,
    increment_revision,
)

# Conditional edge — after validate

def _error_or_research(state: TravelPlanState) -> str:
    return "error" if state.get("status") == "error" else "research"

# Conditional edge — after increment_revision

def _post_increment_route(state: TravelPlanState) -> str:
    action = (state.get("hitl_feedback") or {}).get("action", "approve")
    return "planner" if action == "modify" else "research"

# Build the StateGraph

def _build() -> StateGraph:
    builder = StateGraph(TravelPlanState)

    builder.add_node("validate",           validate_node)
    builder.add_node("research",           run_research_node)
    builder.add_node("planner",            run_planner_node)
    builder.add_node("hitl",               hitl_node)
    builder.add_node("increment_revision", increment_revision)
    builder.add_node("finalize",           finalize_node)
    builder.add_node("error",              error_node)

    builder.add_edge(START, "validate")

    builder.add_conditional_edges(
        "validate",
        _error_or_research,
        {"error": "error", "research": "research"},
    )

    builder.add_edge("research", "planner")
    builder.add_edge("planner",  "hitl")

    builder.add_conditional_edges(
        "hitl",
        revision_router,
        {
            "finalize": "finalize",
            "planner":  "increment_revision",
            "research": "increment_revision",
        },
    )

    # After incrementing: route to the right agent
    builder.add_conditional_edges(
        "increment_revision",
        _post_increment_route,
        {"planner": "planner", "research": "research"},
    )

    builder.add_edge("finalize", END)
    builder.add_edge("error",    END)

    return builder

# Compiled graph — import this in routes.py

graph = _build().compile(checkpointer=checkpointer)

# Config helpers

def make_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}

def get_state(session_id: str) -> dict | None:
    try:
        snap = graph.get_state(make_config(session_id))
        return snap.values if snap and snap.values else None
    except Exception:
        return None

def is_interrupted(session_id: str) -> bool:
    try:
        snap = graph.get_state(make_config(session_id))
        return snap is not None and "hitl" in (snap.next or [])
    except Exception:
        return False
