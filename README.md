# AI Travel Planner

Multi-agent travel planning system with human-in-the-loop approval, built with LangGraph and FastAPI.

## Overview

AI-powered travel assistant that takes user preferences and produces a complete trip plan through a multi-agent workflow. Features pause-and-resume HITL approval across HTTP requests using LangGraph's SqliteSaver checkpointer.

**Assignment:** Express Analytics Take-Home for AI/ML Engineer role.

## Architecture

### System Components

**Orchestrator (LangGraph StateGraph)**
- Manages workflow: validate → research → plan → HITL pause → finalize
- Routes work to specialized agents based on workflow stage
- Handles state transitions and revision loops (max 3 revisions)
- Persists state across HTTP requests via SqliteSaver checkpointer

**Agent 1: Research Agent**
- Gathers destination intelligence via web search and currency conversion
- Tools:
  1. **web_search_tool** (Serper API, mandatory) — Real-time Google search for attractions, safety, visa info, seasonal tips
  2. **currency_converter_tool** (ExchangeRate-API, free tier) — Converts budget to local currency for purchasing power context
- Output: ResearchOutput (Pydantic) with destination overview, attractions, safety notes, visa info, currency context, destination tier

**Agent 2: Itinerary Planner Agent**
- Constructs day-by-day plans using research output
- Tools:
  1. **budget_allocator** (pure Python) — Distributes budget across accommodation/food/activities/transport/contingency by destination tier
  2. **packing_list_generator_tool** (pure Python) — Context-aware packing checklist based on weather, activities, duration
- Output: ItineraryOutput (Pydantic) with daily plan, budget breakdown, packing list, cost validation

**Human-in-the-Loop (HITL)**
- Workflow pauses after draft itinerary generation using `langgraph.types.interrupt()`
- User reviews via `POST /plan/{id}/review` with actions:
  - **approve** — finalize and complete
  - **reject** — re-run research + planning with feedback
  - **modify** — re-run planner only with specific changes
- State persists across pause via SqliteSaver (session_id = thread_id)

### Graph Topology

```
START → validate
  ├─[error]→ error → END
  └─[ok]──→ research
              → planner
                  → hitl (interrupt pause)
                      → [revision_router]
                          ├─ "finalize"  → finalize → END
                          ├─ "planner"   → increment_revision → planner (loop)
                          └─ "research"  → increment_revision → research (loop)
```

## Tools

### Research Agent Tools

| Tool | Type | Purpose | Free Tier |
|------|------|---------|-----------|
| `web_search_tool` | External API (Serper) | Google search results for destination research | 2,500 queries on signup |
| `currency_converter_tool` | External API (ExchangeRate) | Convert USD budget to local currency | 1,500 requests/month |

### Planner Agent Tools

| Tool | Type | Purpose |
|------|------|---------|
| `budget_allocator` | Pure Python | Tier-based budget distribution across 5 categories |
| `packing_list_generator_tool` | Pure Python | Weather + activity-aware packing checklist |

## Setup

### Prerequisites
- Python 3.10+
- API keys: SERPER_API_KEY (serper.dev), EXCHANGERATE_API_KEY (exchangerate-api.com), OPENAI_API_KEY

### Installation

```bash
# Clone repository
git clone <repo-url>
cd AI_Travel_Planner

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set API keys
cp .env.example .env
# Edit .env with your keys:
#   SERPER_API_KEY=<your-key>
#   EXCHANGERATE_API_KEY=<your-key>
#   OPENAI_API_KEY=<your-key>
#   LLM_MODEL=gpt-4o-mini  # or gpt-4o for better quality
```

### Run Application

**Option 1: API Only (FastAPI backend)**

```bash
# Development server
uvicorn main:app --reload --port 8000

# Production server
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

**Option 2: Full Stack (FastAPI + Streamlit UI)**

```bash
# Terminal 1: Start FastAPI backend
uvicorn main:app --reload --port 8000

# Terminal 2: Start Streamlit UI
streamlit run ui.py
```

Navigate to:
- **Streamlit UI:** http://localhost:8501 (beautiful dark theme interface)
- **FastAPI Docs:** http://localhost:8000/docs (interactive API documentation)

### Streamlit UI Features

The `ui.py` provides a production-ready web interface with:

- 🎨 **Dark Theme:** Professional dark mode by default
- 🔄 **Real-Time Streaming:** Watch agents think and work in real-time via SSE
- 🤖 **Agent Activity Log:** See tool calls, LLM thoughts, and workflow transitions
- ⏸️ **HITL Controls:** Approve, reject, or modify plans with interactive forms
- 📋 **Beautiful Itinerary Display:** Expandable daily plans with activities, costs, and tips
- 💾 **Session Management:** Resume previous sessions or start fresh
- 📊 **Live Status Updates:** Track workflow progress (researching → planning → review → complete)
- 📥 **Download Plans:** Export finalized plans as JSON

**UI Screenshot Flow:**
1. Enter destination, dates, budget, interests → Submit
2. Watch real-time agent activity stream (tool calls, thoughts, results)
3. Review draft itinerary when agents pause for HITL
4. Approve/modify/reject with comments
5. Download final plan or start new session

## API Endpoints

Base URL: `http://localhost:8000`

### `POST /plan`
Submit new travel request. Returns session_id for tracking.

**Request:**
```json
{
  "destination": "Tokyo",
  "start_date": "2026-09-01",
  "end_date": "2026-09-07",
  "budget_usd": 3000,
  "travelers": 2,
  "interests": ["culture", "food", "nature"]
}
```

**Response:**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "researching",
  "message": "Travel plan initiated. Researching Tokyo..."
}
```

### `GET /plan/{session_id}`
Get current plan status and draft itinerary (if available).

**Response (researching):**
```json
{
  "session_id": "...",
  "status": "researching",
  "research_output": null,
  "draft_itinerary": null
}
```

**Response (awaiting_review):**
```json
{
  "session_id": "...",
  "status": "awaiting_review",
  "research_output": {...},
  "draft_itinerary": {
    "summary": "7-day cultural exploration of Tokyo featuring Senso-ji Temple, Meiji Shrine, and teamLab Planets",
    "daily_plan": [
      {
        "day": 1,
        "date": "2026-09-01",
        "theme": "Traditional Tokyo",
        "activities": [
          {
            "name": "Senso-ji Temple",
            "time": "9:00 AM - 11:00 AM",
            "duration_hrs": 2.0,
            "cost_per_person_usd": 0,
            "category": "culture",
            "description": "Tokyo's oldest temple in Asakusa district",
            "is_must_see": true
          }
        ],
        "meals": ["Breakfast: Hotel", "Lunch: Tsukiji Market sushi", "Dinner: Ramen in Shinjuku"],
        "accommodation": "Hotel in Shinjuku",
        "daily_cost_per_person_usd": 120.5,
        "notes": "Book temple visit for early morning to avoid crowds"
      }
    ],
    "budget_breakdown": {
      "accommodation": 52.5,
      "food": 37.5,
      "activities": 30.0,
      "transport": 15.0,
      "contingency": 15.0
    },
    "packing_list": {
      "Documents": ["Passport", "Travel insurance", ...],
      "Clothing": ["Lightweight shirts", "Long pants", ...],
      ...
    },
    "budget_status": "within_budget",
    "recommendations": [
      "Buy 7-day Tokyo Metro pass (¥3,000) instead of daily tickets",
      "Many temples are free — save activities budget for teamLab/Skytree"
    ]
  }
}
```

### `POST /plan/{session_id}/review`
Submit HITL feedback after reviewing draft itinerary.

**Request (approve):**
```json
{
  "action": "approve",
  "comments": "Looks great!"
}
```

**Request (reject):**
```json
{
  "action": "reject",
  "comments": "Too focused on culture. Want more nature and outdoor activities."
}
```

**Request (modify):**
```json
{
  "action": "modify",
  "comments": "Swap Day 3 afternoon activity",
  "modified_itinerary": {
    "daily_plan": [
      {
        "day": 3,
        "activities": [...]  // Override Day 3
      }
    ]
  }
}
```

**Response:**
```json
{
  "session_id": "...",
  "status": "planning",  // or "complete" if approved
  "message": "Feedback received. Revising plan..."
}
```

### `GET /plan/{session_id}/final`
Retrieve finalized plan (only after approval).

**Response:**
```json
{
  "session_id": "...",
  "destination": "Tokyo",
  "travel_dates": "2026-09-01 → 2026-09-07",
  "travelers": 2,
  "budget_usd": 3000,
  "itinerary": {...},  // Full ItineraryOutput
  "research_notes": {...},  // Full ResearchOutput
  "approved_at": "2026-06-07T14:32:10Z",
  "revision_count": 1
}
```

## Design Decisions

### Why these 4 tools?

**Research Agent:**
- **Serper over Exa:** 2,500 free queries, simpler REST API, returns organic Google results (no neural search ambiguity)
- **Currency converter:** Directly relevant to budget planning. Every good repo in the wild uses this. Shows systems thinking — budget context matters for international travel.

**Planner Agent:**
- **Budget allocator over flight/hotel APIs:** Zero external dependency = zero failure mode during demo. Pure logic that evaluates immediately understand. Shows cost-tier awareness (budget/mid/luxury).
- **Packing list over restaurant recommender:** Assignment explicitly mentions it as example. Context-aware (weather + activities) adds real user value. No API rate limits.

### Why SqliteSaver over InMemoryCheckpointer?

HITL pause-and-resume across HTTP requests requires durable storage. InMemoryCheckpointer loses state on server restart. SqliteSaver persists to `travel_planner.db` — simple, local, no Postgres setup for demo.

### Why no long-term memory (InMemoryStore) in final implementation?

Assignment doesn't require cross-session personalization. Removed `get_user_travel_history` and `save_destination_preference` tools to stay focused on core workflow: one request → research → plan → HITL → finalize.

### Why structured output (response_format)?

LangGraph docs pattern. Pydantic schemas (ResearchOutput, ItineraryOutput) enforce consistent structure. Auto-selects ProviderStrategy for OpenAI models — no manual parsing, retry on schema mismatch.

## Trade-offs

| Decision | Benefit | Cost |
|----------|---------|------|
| Pure Python tools (budget, packing) | Zero API failure risk | Less dynamic than real-time APIs |
| Web search for attractions vs. POI database | Real-time, destination-agnostic | API cost, rate limits |
| SqliteSaver checkpointer | Simple local setup | Not production-ready (use PostgresCheckpointer in prod) |
| Max 3 revisions | Prevents infinite loops | User might need to start new session |
| gpt-4o-mini default | Fast, cheap ($0.15/1M tokens) | Lower quality than gpt-4o — tune via LLM_MODEL env var |

## Production Improvements (Given More Time)

**Infrastructure:**
- Swap SqliteSaver → PostgresCheckpointer for multi-instance deployments
- Add Redis cache for Serper results (reduce API calls on revisions)
- Rate limiting on `/plan` endpoint (prevent API key exhaustion)
- Structured logging (JSON) with request_id tracing

**Tool Enhancements:**
- Replace mock weather in packing_list_generator with OpenWeatherMap forecast API
- Add Google Places API for real POI data (costs, hours, reviews)
- Flight/hotel price estimation via Amadeus API (requires paid tier)

**Agent Improvements:**
- Multi-model strategy: gpt-4o for research (quality), gpt-4o-mini for planning (speed)
- Retry logic with exponential backoff on API failures
- Parallel tool execution (call web_search + currency_converter simultaneously)

**User Experience:**
- Streaming progress updates via Server-Sent Events (`GET /plan/{id}/stream`)
- Itinerary visualization (map with pins, timeline view)
- Export to PDF/Google Calendar
- Multi-destination support (Tokyo → Kyoto → Osaka in one plan)

**Testing:**
- Unit tests for tools (mock Serper/ExchangeRate responses)
- Integration tests for full workflow (validate → research → plan → HITL → finalize)
- Load testing (concurrent sessions, checkpointer contention)

## References

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Serper API](https://serper.dev)
- [ExchangeRate-API](https://www.exchangerate-api.com)
- [FastAPI Documentation](https://fastapi.tiangolo.com)

---

**Author:** Manoj Kumar Karumanchi  
**Assignment:** Express Analytics AI/ML Engineer Take-Home  
**Date:** June 2026ggraph>
