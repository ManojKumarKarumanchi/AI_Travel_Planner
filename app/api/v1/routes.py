
from __future__ import annotations

import asyncio
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from langgraph.types import Command

from app.services.state import TravelRequest, HITLFeedback
from app.services.workflow import graph, make_config, get_state, is_interrupted
from app.flow.visualize import generate_session_graph
from app.utils.logger import get_logger

router   = APIRouter(prefix="/plan", tags=["Travel Planner"])
_executor = ThreadPoolExecutor(max_workers=4)
logger = get_logger("api.routes")

# Pydantic request/response models

class PlanRequest(BaseModel):
    origin:      str  = Field(..., min_length=2, example="New York, USA", description="Origin city")
    destination: str  = Field(..., min_length=2, example="Tokyo, Japan", description="Destination city")
    start_date:  str  = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", example="2025-09-10")
    end_date:    str  = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", example="2025-09-17")
    budget_usd:  int  = Field(..., gt=0, example=3000)
    travelers:   int  = Field(..., gt=0, le=20, example=2)
    interests:   list[str] = Field(default_factory=list, example=["food", "culture"])
    comments:    Optional[str] = Field(default=None, example="Vegetarian meals preferred")
    user_id:     str  = Field(default="anonymous", example="user_123")

    @field_validator("interests", mode="before")
    @classmethod
    def _norm(cls, v):
        return [i.lower().strip() for i in v if i] if isinstance(v, list) else v

class ReviewRequest(BaseModel):
    action:             str = Field(..., pattern="^(approve|reject|modify)$")
    comments:           Optional[str] = None
    modified_itinerary: Optional[dict] = None

class PlanStatusResponse(BaseModel):
    session_id:      str
    status:          str
    awaiting_review: bool
    draft_itinerary: Optional[dict]
    error_message:   Optional[str]

class FinalPlanResponse(BaseModel):
    session_id: str
    final_plan:  dict

async def _run_in_thread(fn, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))

# POST /plan — submit a new travel request

@router.post("", status_code=status.HTTP_202_ACCEPTED,
             summary="Submit a new travel planning request")
async def create_plan(body: PlanRequest) -> dict:
    session_id = str(uuid.uuid4())
    logger.info(f"[CREATE_PLAN] New request - Session: {session_id}, Destination: {body.destination}")
    config     = make_config(session_id)

    destination_full = f"{body.destination} (from {body.origin})"
    if body.comments:
        destination_full += f" - {body.comments}"

    request_data = TravelRequest(
        destination=destination_full,
        start_date=body.start_date,
        end_date=body.end_date,
        budget_usd=body.budget_usd,
        travelers=body.travelers,
        interests=body.interests,
        # Passed through to context= in agent nodes
    )
    request_dict = dict(request_data)
    request_dict["user_id"] = body.user_id
    request_dict["origin"] = body.origin
    request_dict["comments"] = body.comments

    initial_state = {
        "session_id":      session_id,
        "request":         request_dict,
        "messages":        [],
        "research_output": None,
        "draft_itinerary": None,
        "hitl_feedback":   None,
        "final_plan":      None,
        "status":          "validating",
        "revision_count":  0,
        "error_message":   None,
    }

    logger.info(f"[CREATE_PLAN] Starting graph execution - Session: {session_id}")
    result = await _run_in_thread(graph.invoke, initial_state, config)
    logger.info(f"[CREATE_PLAN] Graph returned - Session: {session_id}, Status: {result.get('status')}")

    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=result.get("error_message", "Validation failed"),
        )

    return {
        "session_id": session_id,
        "status":     result.get("status"),
        "message":    "Poll GET /plan/{session_id} until status is 'awaiting_review'.",
    }

# GET /plan/{session_id} — poll current status

@router.get("/{session_id}", response_model=PlanStatusResponse,
            summary="Get current plan status and draft itinerary")
async def get_plan(session_id: str) -> PlanStatusResponse:
    state = get_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return PlanStatusResponse(
        session_id=session_id,
        status=state.get("status", "unknown"),
        awaiting_review=is_interrupted(session_id),
        draft_itinerary=state.get("draft_itinerary"),
        error_message=state.get("error_message"),
    )

@router.post("/{session_id}/review",
             summary="Submit human review: approve / reject / modify")
async def review_plan(session_id: str, body: ReviewRequest) -> dict:
    state = get_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    if not is_interrupted(session_id):
        raise HTTPException(
            status_code=409,
            detail=f"Session not awaiting review. Current status: {state.get('status')}.",
        )

    feedback = {
        "action":             body.action,
        "comments":           body.comments,
        "modified_itinerary": body.modified_itinerary,
    }
    config = make_config(session_id)
    result = await _run_in_thread(graph.invoke, Command(resume=feedback), config)

    return {
        "session_id": session_id,
        "action":     body.action,
        "status":     result.get("status"),
        "message":    _review_message(body.action, result.get("status", "")),
    }

def _review_message(action: str, cur_status: str) -> str:
    if action == "approve":
        return "Plan approved and finalised." if cur_status == "complete" else f"Processing… status: {cur_status}"
    if action == "reject":
        return "Rejected — re-researching and replanning. Poll GET /plan/{id}."
    if action == "modify":
        return "Modifications accepted — replanning. Poll GET /plan/{id}."
    return f"Status: {cur_status}"

@router.get("/{session_id}/final", response_model=FinalPlanResponse,
            summary="Retrieve the finalised travel plan (after approval)")
async def get_final_plan(session_id: str) -> FinalPlanResponse:
    state = get_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    if state.get("status") != "complete":
        raise HTTPException(
            status_code=409,
            detail=f"Plan not yet finalised. Status: {state.get('status')}. Approve first.",
        )
    return FinalPlanResponse(session_id=session_id, final_plan=state["final_plan"])

#
#
# SSE format: "data: {json}\n\n"

@router.get("/{session_id}/stream",
            summary="SSE stream — live tokens from research and planning agents")
async def stream_plan(session_id: str) -> StreamingResponse:
    state = get_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    async def _sse_generator() -> AsyncIterator[str]:
        config = make_config(session_id)
        current_node = None

        # Send initial heartbeat
        yield f"data: {json.dumps({'type': 'connected', 'session_id': session_id})}\n\n"

        try:
            async for chunk in graph.astream(
                None,           # None = resume from latest checkpoint
                config=config,
                stream_mode=["messages", "updates", "values"],
            ):
                if isinstance(chunk, tuple) and len(chunk) == 2:
                    msg, meta = chunk
                    token_text = getattr(msg, "content", "")
                    if token_text:
                        yield f"data: {json.dumps({'type': 'token', 'content': token_text, 'node': current_node})}\n\n"

                    # Tool call detection (from AIMessage.tool_calls)
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for tool_call in msg.tool_calls:
                            yield f"data: {json.dumps({'type': 'tool_call', 'tool': tool_call.get('name'), 'args': tool_call.get('args'), 'node': current_node})}\n\n"

                elif isinstance(chunk, dict):
                    for node_name in chunk:
                        if node_name in ("research", "planner"):
                            current_node = node_name
                            yield f"data: {json.dumps({'type': 'agent_start', 'node': node_name})}\n\n"

                        node_data = chunk[node_name]

                        if node_data and "messages" in node_data:
                            for msg in node_data["messages"]:
                                if hasattr(msg, "type") and msg.type == "tool":
                                    tool_name = getattr(msg, "name", "unknown")
                                    tool_output = getattr(msg, "content", "")[:200]  # truncate for SSE
                                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': tool_name, 'output': tool_output, 'node': current_node})}\n\n"

                        # Emit agent_end when leaving research/planner
                        if node_name in ("research", "planner") and node_data:
                            yield f"data: {json.dumps({'type': 'agent_end', 'node': node_name})}\n\n"
                            current_node = None

                        # Detect interrupt payload
                        if "__interrupt__" in (node_data or {}):
                            interrupt_val = node_data["__interrupt__"]
                            yield f"data: {json.dumps({'type': 'interrupt', 'payload': interrupt_val})}\n\n"

                        # Detect completion
                        if (node_data or {}).get("status") == "complete":
                            yield f"data: {json.dumps({'type': 'done', 'status': 'complete'})}\n\n"
                            return

                # Send periodic heartbeat to keep connection alive
                await asyncio.sleep(0.1)  # Small delay between chunks

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc), 'node': current_node})}\n\n"

        finally:
            # Send final completion message
            yield f"data: {json.dumps({'type': 'stream_end'})}\n\n"

    return StreamingResponse(
        _sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":       "no-cache",
            "X-Accel-Buffering":   "no",    # disable nginx buffering
            "Connection":          "keep-alive",
            "Keep-Alive":          "timeout=600",  # 10 minutes
        },
    )

@router.get("/{session_id}/graph",
            summary="Get workflow graph visualization for this session")
async def get_session_graph(session_id: str):
    from fastapi.responses import FileResponse

    state = get_state(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

    try:
        # Generate session-specific graph
        graph_path = generate_session_graph(graph, session_id, state)

        if graph_path and os.path.exists(graph_path):
            return FileResponse(
                graph_path,
                media_type="image/png",
                filename=f"workflow_{session_id[:8]}.png"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail="Graph visualization not available. Install pygraphviz: pip install pygraphviz"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Graph generation failed: {str(e)}")
