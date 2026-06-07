"""
ui.py — Streamlit frontend for AI Travel Planner.

Run alongside FastAPI backend:
    Terminal 1: uvicorn main:app --reload --port 8000
    Terminal 2: streamlit run ui.py

Features:
  - Dark theme by default
  - Real-time streaming of agent thoughts + tool calls
  - Session management
  - HITL pause/resume controls
  - Beautiful itinerary rendering
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import Optional
import sseclient  # pip install sseclient-py

# Configuration

API_BASE_URL = "http://localhost:8000"

# Dark theme configuration
st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for dark theme enhancement
st.markdown("""
<style>
    /* Dark theme enhancements */
    .main {
        background-color: #0e1117;
    }

    /* Agent activity box */
    .agent-box {
        background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #60a5fa;
    }

    /* Tool call box */
    .tool-box {
        background: linear-gradient(135deg, #065f46 0%, #047857 100%);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
        border-left: 3px solid #34d399;
        font-family: 'Courier New', monospace;
    }

    /* Status indicators */
    .status-researching { color: #60a5fa; }
    .status-planning { color: #a78bfa; }
    .status-review { color: #fbbf24; }
    .status-complete { color: #34d399; }
    .status-error { color: #f87171; }

    /* Streaming token animation */
    .streaming-token {
        animation: pulse 1.5s ease-in-out infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Session ID badge */
    .session-badge {
        background: #374151;
        padding: 5px 10px;
        border-radius: 5px;
        font-family: monospace;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# Session State Initialization

if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "plan_status" not in st.session_state:
    st.session_state.plan_status = None

if "draft_itinerary" not in st.session_state:
    st.session_state.draft_itinerary = None

if "streaming_logs" not in st.session_state:
    st.session_state.streaming_logs = []

if "current_agent" not in st.session_state:
    st.session_state.current_agent = None

# API Client Functions

def check_api_health():
    """Check if FastAPI backend is running."""
    try:
        resp = requests.get(f"{API_BASE_URL}/health", timeout=3)
        return resp.status_code == 200, resp.json()
    except Exception as e:
        return False, {"error": str(e)}

def create_travel_plan(origin, destination, start_date, end_date, budget, travelers, interests, comments=None):
    """Submit new travel plan request."""
    payload = {
        "origin": origin,
        "destination": destination,
        "start_date": start_date,
        "end_date": end_date,
        "budget_usd": budget,
        "travelers": travelers,
        "interests": interests,
        "comments": comments,
        "user_id": "streamlit_user"
    }

    try:
        resp = requests.post(f"{API_BASE_URL}/plan", json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.HTTPError as e:
        st.error(f"API Error: {e.response.json().get('detail', str(e))}")
        return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def get_plan_status(session_id):
    """Poll plan status."""
    try:
        resp = requests.get(f"{API_BASE_URL}/plan/{session_id}", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to get plan status: {str(e)}")
        return None

def submit_review(session_id, action, comments=None, modifications=None):
    """Submit HITL review decision."""
    payload = {
        "action": action,
        "comments": comments,
        "modified_itinerary": modifications
    }

    try:
        resp = requests.post(f"{API_BASE_URL}/plan/{session_id}/review", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Review submission failed: {str(e)}")
        return None

def get_final_plan(session_id):
    """Retrieve finalized plan."""
    try:
        resp = requests.get(f"{API_BASE_URL}/plan/{session_id}/final", timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"Failed to get final plan: {str(e)}")
        return None

def stream_events(session_id):
    """
    Stream SSE events from /plan/{id}/stream endpoint.

    Yields event dicts: {"type": "token", "content": "..."}
    """
    try:
        url = f"{API_BASE_URL}/plan/{session_id}/stream"
        response = requests.get(url, stream=True, timeout=600)  # 10 minutes
        response.raise_for_status()

        client = sseclient.SSEClient(response)
        for event in client.events():
            if event.data:
                try:
                    yield json.loads(event.data)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        st.error(f"Streaming error: {str(e)}")

# UI Components

def render_header():
    """Render page header with API status."""
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.title("✈️ AI Travel Planner")
        st.caption("Multi-agent orchestration with human-in-the-loop approval")

    with col2:
        # API Health Check
        is_healthy, health_data = check_api_health()
        if is_healthy:
            st.success("🟢 API Online")
            if health_data.get("warnings"):
                st.warning(f"⚠️ {len(health_data['warnings'])} config warnings")
        else:
            st.error("🔴 API Offline")
            st.caption("Start: `uvicorn main:app --reload`")

    with col3:
        if st.session_state.session_id:
            st.markdown(f"""
            <div class="session-badge">
                Session: {st.session_state.session_id[:8]}...
            </div>
            """, unsafe_allow_html=True)

def render_input_form():
    """Render travel plan input form with better UX."""
    st.header("📋 Plan Your Trip")

    # Use form for better UX
    with st.form("travel_plan_form", clear_on_submit=False):
        st.subheader("✈️ Trip Details")

        # Row 1: From/To Places (side by side)
        col_from, col_to = st.columns(2)

        with col_from:
            origin = st.text_input(
                "🏠 From (Origin City) *",
                placeholder="e.g., New York, London, Mumbai",
                help="Where are you traveling from?",
                key="origin_input"
            )

        with col_to:
            destination = st.text_input(
                "🌍 To (Destination) *",
                placeholder="e.g., Tokyo, Paris, Bali",
                help="Where do you want to go?",
                key="dest_input"
            )

        # Row 2: Dates (side by side)
        col_date1, col_date2 = st.columns(2)

        with col_date1:
            st.markdown("**📅 From (Start Date)** *")
            start_date = st.date_input(
                "Departure date",
                min_value=datetime.now().date(),
                help="Format: YYYY-MM-DD | When does your trip start?",
                label_visibility="collapsed",
                key="start_input"
            )
            st.caption(f"Selected: {start_date.strftime('%Y-%m-%d (%A)')}")

        with col_date2:
            st.markdown("**📅 To (End Date)** *")
            end_date = st.date_input(
                "Return date",
                min_value=datetime.now().date(),
                help="Format: YYYY-MM-DD | When does your trip end?",
                label_visibility="collapsed",
                key="end_input"
            )
            st.caption(f"Selected: {end_date.strftime('%Y-%m-%d (%A)')}")

        # Calculate trip duration
        if end_date > start_date:
            duration = (end_date - start_date).days
            st.info(f"📆 Trip Duration: **{duration} days**")
        elif end_date <= start_date:
            st.warning("⚠️ End date must be after start date")

        st.markdown("---")

        # Row 3: Budget and Travelers (side by side)
        col_budget, col_travelers = st.columns(2)

        with col_budget:
            budget = st.number_input(
                "💰 Total Budget (USD) *",
                min_value=100,
                max_value=100000,
                value=3000,
                step=100,
                help="Total budget for ALL travelers combined",
                key="budget_input"
            )
            if budget and st.session_state.get("travelers_input", 2) > 0:
                per_person = budget / st.session_state.get("travelers_input", 2)
                st.caption(f"≈ ${per_person:,.0f} per person")

        with col_travelers:
            travelers = st.number_input(
                "👥 Number of Travelers *",
                min_value=1,
                max_value=20,
                value=2,
                help="How many people are traveling?",
                key="travelers_input"
            )
            st.caption(f"Planning for {travelers} {'person' if travelers == 1 else 'people'}")

        st.markdown("---")

        # Row 4: Interests
        st.markdown("**🎯 What are you interested in?** *")
        interests = st.multiselect(
            "Select all that apply",
            options=[
                "culture",
                "food",
                "nature",
                "adventure",
                "shopping",
                "nightlife",
                "history",
                "beach",
                "hiking",
                "art",
                "museums",
                "photography",
                "wildlife",
                "wellness"
            ],
            default=["culture", "food"],
            help="Select multiple interests to personalize your itinerary",
            label_visibility="collapsed",
            key="interests_input"
        )

        if not interests:
            st.caption("💡 Tip: Select at least one interest for better recommendations")

        st.markdown("---")

        # Row 5: Additional Comments/Notes
        st.markdown("**💬 Additional Notes or Special Requests** (optional)")
        comments = st.text_area(
            "Any specific requirements, preferences, or comments?",
            placeholder="e.g., 'Vegetarian meals only', 'Need wheelchair accessible places', 'Avoid early morning activities', 'Celebrating anniversary'",
            help="Optional: Add any special requests, dietary restrictions, accessibility needs, or preferences",
            label_visibility="collapsed",
            key="comments_input",
            height=80
        )

        st.markdown("---")

        # Submit buttons
        col_submit, col_clear, col_spacer = st.columns([2, 2, 3])

        with col_submit:
            submit = st.form_submit_button(
                "🚀 Create Travel Plan",
                type="primary",
                use_container_width=True
            )

        with col_clear:
            clear = st.form_submit_button(
                "🗑️ Clear Form",
                use_container_width=True
            )

    # Handle clear button
    if clear:
        st.session_state.session_id = None
        st.session_state.plan_status = None
        st.session_state.draft_itinerary = None
        st.session_state.streaming_logs = []
        st.rerun()

    # Handle submit button
    if submit:
        # Validation
        errors = []

        if not origin or len(origin.strip()) < 2:
            errors.append("❌ Please enter origin city")

        if not destination or len(destination.strip()) < 2:
            errors.append("❌ Please enter destination")

        if not start_date:
            errors.append("❌ Please select a start date")

        if not end_date:
            errors.append("❌ Please select an end date")

        if start_date and end_date and end_date <= start_date:
            errors.append("❌ End date must be after start date")

        if budget < 100:
            errors.append("❌ Budget must be at least $100")

        if travelers < 1:
            errors.append("❌ Number of travelers must be at least 1")

        if not interests:
            errors.append("❌ Please select at least one interest")

        # Show validation errors
        if errors:
            for error in errors:
                st.error(error)
            return

        # All validation passed - submit to API
        with st.spinner("🔄 Submitting your travel plan request..."):
            # Prepare payload with separate origin and destination
            payload = {
                "origin": origin.strip(),
                "destination": destination.strip(),
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "budget_usd": budget,
                "travelers": travelers,
                "interests": interests,
                "comments": comments.strip() if comments else None,
                "user_id": "streamlit_user"
            }

            result = create_travel_plan(
                origin=payload["origin"],
                destination=payload["destination"],
                start_date=payload["start_date"],
                end_date=payload["end_date"],
                budget=payload["budget_usd"],
                travelers=payload["travelers"],
                interests=payload["interests"],
                comments=payload["comments"]
            )

            if result and "session_id" in result:
                st.session_state.session_id = result["session_id"]
                st.session_state.streaming_logs = []
                st.session_state.user_comments = comments
                st.session_state.plan_status = "researching"  # Set initial status

                st.success("Plan request submitted! Connecting to live stream...")
                st.info(f"Session ID: `{result['session_id'][:16]}...`")

                # Show summary
                st.markdown("**Request Summary:**")
                st.markdown(f"""
                - From: {origin} → To: {destination}
                - Dates: {start_date.strftime('%b %d, %Y')} → {end_date.strftime('%b %d, %Y')} ({(end_date - start_date).days} days)
                - Budget: ${budget:,} USD (${budget/travelers:,.0f}/person)
                - Travelers: {travelers}
                - Interests: {', '.join(interests)}
                {f"- Notes: {comments}" if comments else ""}
                """)

                st.markdown("**Connecting to agent stream...**")
                time.sleep(1)
                st.rerun()

def render_streaming_activity():
    """Render real-time streaming activity log."""
    st.header("🔄 Agent Activity Stream")

    if not st.session_state.session_id:
        st.info("Submit a travel plan to see real-time agent activity")
        return

    # Container for streaming logs
    log_container = st.container()
    status_placeholder = st.empty()

    with log_container:
        # Auto-scroll container with fixed height
        st.markdown('<div style="height: 400px; overflow-y: auto; background: #1e1e1e; padding: 15px; border-radius: 10px;">', unsafe_allow_html=True)

        log_display = st.empty()

        # Stream events
        for event in stream_events(st.session_state.session_id):
            event_type = event.get("type")

            # Update current agent
            if event_type == "agent_start":
                st.session_state.current_agent = event.get("node")
                st.session_state.streaming_logs.append({
                    "type": "agent_start",
                    "agent": event.get("node"),
                    "time": datetime.now().strftime("%H:%M:%S")
                })

            elif event_type == "tool_call":
                st.session_state.streaming_logs.append({
                    "type": "tool_call",
                    "tool": event.get("tool"),
                    "args": event.get("args"),
                    "time": datetime.now().strftime("%H:%M:%S")
                })

            elif event_type == "tool_result":
                st.session_state.streaming_logs.append({
                    "type": "tool_result",
                    "tool": event.get("tool"),
                    "output": event.get("output"),
                    "time": datetime.now().strftime("%H:%M:%S")
                })

            elif event_type == "token":
                # Append token to last log entry or create new
                if st.session_state.streaming_logs and st.session_state.streaming_logs[-1].get("type") == "token":
                    st.session_state.streaming_logs[-1]["content"] += event.get("content", "")
                else:
                    st.session_state.streaming_logs.append({
                        "type": "token",
                        "content": event.get("content", ""),
                        "agent": event.get("node"),
                        "time": datetime.now().strftime("%H:%M:%S")
                    })

            elif event_type == "agent_end":
                st.session_state.streaming_logs.append({
                    "type": "agent_end",
                    "agent": event.get("node"),
                    "time": datetime.now().strftime("%H:%M:%S")
                })
                st.session_state.current_agent = None

            elif event_type == "interrupt":
                st.session_state.streaming_logs.append({
                    "type": "interrupt",
                    "payload": event.get("payload"),
                    "time": datetime.now().strftime("%H:%M:%S")
                })
                break  # Stop streaming, waiting for HITL

            elif event_type == "done":
                st.session_state.streaming_logs.append({
                    "type": "done",
                    "time": datetime.now().strftime("%H:%M:%S")
                })
                break

            elif event_type == "error":
                st.session_state.streaming_logs.append({
                    "type": "error",
                    "message": event.get("message"),
                    "time": datetime.now().strftime("%H:%M:%S")
                })
                break

            # Render logs
            render_logs(log_display)

        st.markdown('</div>', unsafe_allow_html=True)

    # Poll for status updates
    if st.session_state.session_id:
        status_data = get_plan_status(st.session_state.session_id)
        if status_data:
            st.session_state.plan_status = status_data.get("status")
            st.session_state.draft_itinerary = status_data.get("draft_itinerary")

            # Show status badge
            status = status_data.get("status", "unknown")
            status_class = f"status-{status.replace('_', '-')}"
            status_placeholder.markdown(f"""
            <div style="text-align: center; padding: 10px;">
                <span class="{status_class}" style="font-size: 18px; font-weight: bold;">
                    Status: {status.upper().replace('_', ' ')}
                </span>
            </div>
            """, unsafe_allow_html=True)

def render_logs(placeholder):
    """Render streaming logs with formatting."""
    log_html = ""

    for log in st.session_state.streaming_logs[-50:]:  # Show last 50 entries
        log_type = log.get("type")
        time_str = log.get("time", "")

        if log_type == "agent_start":
            agent = log.get("agent", "unknown")
            log_html += f"""
            <div class="agent-box">
                <strong>🤖 {agent.upper()} AGENT STARTED</strong>
                <span style="float: right; opacity: 0.7;">{time_str}</span>
            </div>
            """

        elif log_type == "tool_call":
            tool = log.get("tool", "")
            args = log.get("args", {})
            log_html += f"""
            <div class="tool-box">
                <strong>🔧 TOOL CALL:</strong> {tool}<br>
                <small style="opacity: 0.8;">Args: {json.dumps(args, indent=2)}</small>
                <span style="float: right; opacity: 0.7;">{time_str}</span>
            </div>
            """

        elif log_type == "tool_result":
            tool = log.get("tool", "")
            output = log.get("output", "")[:150]
            log_html += f"""
            <div class="tool-box" style="opacity: 0.8;">
                <strong>✓ RESULT:</strong> {tool}<br>
                <small>{output}...</small>
                <span style="float: right; opacity: 0.7;">{time_str}</span>
            </div>
            """

        elif log_type == "token":
            content = log.get("content", "")
            agent = log.get("agent", "")
            log_html += f"""
            <div style="padding: 5px; margin: 3px 0; background: #2d2d2d; border-radius: 5px;">
                <span class="streaming-token" style="color: #a0a0a0;">💭 {content}</span>
            </div>
            """

        elif log_type == "agent_end":
            agent = log.get("agent", "")
            log_html += f"""
            <div class="agent-box" style="opacity: 0.7;">
                <strong>✅ {agent.upper()} AGENT COMPLETED</strong>
                <span style="float: right;">{time_str}</span>
            </div>
            """

        elif log_type == "interrupt":
            log_html += f"""
            <div style="background: #f59e0b; color: #000; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <strong>⏸️ HUMAN REVIEW REQUIRED</strong>
                <span style="float: right;">{time_str}</span>
            </div>
            """

        elif log_type == "done":
            log_html += f"""
            <div style="background: #10b981; color: #000; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <strong>🎉 WORKFLOW COMPLETED</strong>
                <span style="float: right;">{time_str}</span>
            </div>
            """

        elif log_type == "error":
            msg = log.get("message", "")
            log_html += f"""
            <div style="background: #ef4444; color: #fff; padding: 15px; border-radius: 10px; margin: 10px 0;">
                <strong>❌ ERROR:</strong> {msg}
                <span style="float: right;">{time_str}</span>
            </div>
            """

    placeholder.markdown(log_html, unsafe_allow_html=True)

def render_draft_itinerary():
    """Render draft itinerary for review."""
    if not st.session_state.draft_itinerary:
        return

    st.header("📝 Draft Itinerary - Review Required")

    draft = st.session_state.draft_itinerary

    # Summary
    st.subheader(draft.get("summary", "Trip Summary"))

    # Budget Status
    budget_status = draft.get("budget_status", "unknown")
    budget_colors = {
        "within_budget": "🟢",
        "under_budget": "🔵",
        "over_budget": "🔴"
    }
    st.markdown(f"**Budget Status:** {budget_colors.get(budget_status, '⚪')} {budget_status.replace('_', ' ').title()}")
    st.markdown(f"**Total Estimated Cost:** ${draft.get('total_estimated_cost_usd', 0):,.2f}")

    # Daily Plan
    st.markdown("---")
    st.subheader("📅 Day-by-Day Itinerary")

    for day in draft.get("daily_plan", []):
        with st.expander(f"**Day {day.get('day', '?')}: {day.get('theme', 'Activities')}** - {day.get('date', 'TBD')}", expanded=True):
            daily_cost = day.get('daily_cost_per_person_usd', 0)
            try:
                daily_cost_float = float(daily_cost) if daily_cost else 0.0
                st.markdown(f"**Daily Cost:** ${daily_cost_float:.2f} per person")
            except (ValueError, TypeError):
                st.markdown(f"**Daily Cost:** {daily_cost} per person")

            # Activities
            st.markdown("**Activities:**")
            for activity in day.get("activities", []):
                must_see = "[MUST-SEE]" if activity.get("is_must_see") else ""
                try:
                    cost = float(activity.get('cost_per_person_usd', 0))
                    duration = float(activity.get('duration_hrs', 0))
                    st.markdown(f"""
                    - {must_see} **{activity.get('name', 'Activity')}** ({activity.get('category', 'General')})
                      Time: {activity.get('time', 'TBD')} | Cost: ${cost:.2f}/person | Duration: {duration:.1f}hrs
                      _{activity.get('description', '')}_
                    """)
                except (ValueError, TypeError, KeyError):
                    st.markdown(f"- {activity}")

            # Meals
            st.markdown("**Meals:**")
            for meal in day.get("meals", []):
                st.markdown(f"- {meal}")

            # Accommodation
            st.markdown(f"**🏨 Accommodation:** {day.get('accommodation', 'TBD')}")

            # Notes
            if day.get("notes"):
                st.info(f"💡 **Tip:** {day['notes']}")

    # Budget Breakdown
    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Budget Breakdown (per person/day)")
        breakdown = draft.get("budget_breakdown", {})
        for category, amount in breakdown.items():
            try:
                amount_float = float(amount) if isinstance(amount, (str, int)) else amount
                st.markdown(f"- **{category.title()}:** ${amount_float:.2f}")
            except (ValueError, TypeError):
                st.markdown(f"- **{category.title()}:** {amount}")

    with col2:
        st.subheader("🎒 Packing List")
        packing = draft.get("packing_list", {})
        for category, items in packing.items():
            with st.expander(f"**{category}**"):
                for item in items:
                    st.markdown(f"- {item}")

    # Recommendations
    st.markdown("---")
    st.subheader("💡 Pro Tips")
    for rec in draft.get("recommendations", []):
        st.markdown(f"- {rec}")

def render_review_controls():
    """Render HITL review controls (approve/reject/modify)."""
    if st.session_state.plan_status != "awaiting_review":
        return

    st.markdown("---")
    st.header("✋ Your Review")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("✅ Approve Plan", type="primary", use_container_width=True):
            with st.spinner("Approving plan..."):
                result = submit_review(st.session_state.session_id, "approve")
                if result:
                    st.success("Plan approved! Finalizing...")
                    time.sleep(2)
                    st.rerun()

    with col2:
        if st.button("🔄 Request Changes", use_container_width=True):
            st.session_state.show_modify_form = True

    with col3:
        if st.button("❌ Reject & Restart", use_container_width=True):
            st.session_state.show_reject_form = True

    # Modify form
    if getattr(st.session_state, "show_modify_form", False):
        with st.form("modify_form"):
            st.subheader("📝 Modification Request")
            comments = st.text_area(
                "What would you like to change?",
                placeholder="e.g., Day 2 lunch is too expensive, want a cheaper option",
                height=100
            )

            submit_modify = st.form_submit_button("Submit Changes", type="primary")

            if submit_modify:
                if comments:
                    with st.spinner("Submitting modifications..."):
                        result = submit_review(st.session_state.session_id, "modify", comments=comments)
                        if result:
                            st.success("Modifications submitted! Agent is replanning...")
                            st.session_state.show_modify_form = False
                            st.session_state.streaming_logs = []
                            time.sleep(2)
                            st.rerun()
                else:
                    st.error("Please provide modification details")

    # Reject form
    if getattr(st.session_state, "show_reject_form", False):
        with st.form("reject_form"):
            st.subheader("❌ Rejection Feedback")
            comments = st.text_area(
                "Why are you rejecting this plan?",
                placeholder="e.g., Too focused on culture, want more outdoor activities",
                height=100
            )

            submit_reject = st.form_submit_button("Confirm Rejection", type="primary")

            if submit_reject:
                if comments:
                    with st.spinner("Rejecting and restarting..."):
                        result = submit_review(st.session_state.session_id, "reject", comments=comments)
                        if result:
                            st.success("Plan rejected! Starting fresh research and planning...")
                            st.session_state.show_reject_form = False
                            st.session_state.streaming_logs = []
                            time.sleep(2)
                            st.rerun()
                else:
                    st.error("Please provide rejection reason")

def render_final_plan():
    """Render approved final plan."""
    if st.session_state.plan_status != "complete":
        return

    with st.spinner("Loading final plan..."):
        final_plan_data = get_final_plan(st.session_state.session_id)

    if not final_plan_data:
        return

    final = final_plan_data.get("final_plan", {})

    st.success("🎉 Your travel plan is ready!")
    st.header("✈️ Final Travel Plan")

    # Plan metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Destination", final.get("destination"))
    with col2:
        st.metric("Travel Dates", final.get("travel_dates"))
    with col3:
        st.metric("Travelers", final.get("travelers"))

    st.metric("Total Budget", f"${final.get('budget_usd', 0):,}")

    # Render itinerary (same as draft)
    st.session_state.draft_itinerary = final.get("itinerary")
    render_draft_itinerary()

    # Download button
    st.markdown("---")
    plan_json = json.dumps(final, indent=2)
    st.download_button(
        label="📥 Download Plan (JSON)",
        data=plan_json,
        file_name=f"travel_plan_{final.get('destination', 'trip').lower().replace(' ', '_')}.json",
        mime="application/json"
    )

# Main App

def main():
    render_header()

    st.markdown("---")

    # Layout: Input form on left, activity stream on right
    col_input, col_stream = st.columns([1, 1])

    with col_input:
        render_input_form()

        # Show review controls if awaiting review
        if st.session_state.plan_status == "awaiting_review":
            render_review_controls()

    with col_stream:
        if st.session_state.session_id:
            render_streaming_activity()

    # Full-width sections
    if st.session_state.plan_status == "awaiting_review":
        st.markdown("---")
        render_draft_itinerary()

    if st.session_state.plan_status == "complete":
        st.markdown("---")
        render_final_plan()

if __name__ == "__main__":
    main()
