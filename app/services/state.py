
from __future__ import annotations

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from pydantic import field_validator

from langchain.agents import AgentState          # base class from docs
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class TravelRequest(TypedDict):
    destination: str          # Combined: "Tokyo (from New York) - Notes"
    start_date:  str          # ISO-8601  "2025-09-01"
    end_date:    str
    budget_usd:  int
    travelers:   int
    interests:   list[str]
    origin:      Optional[str]     # Origin city (stored separately for reference)
    comments:    Optional[str]     # User notes (embedded in destination for LLM)

# HITL feedback schema

class HITLFeedback(TypedDict):
    action:             Literal["approve", "reject", "modify"]
    comments:           Optional[str]
    modified_itinerary: Optional[dict]

from pydantic import BaseModel, Field

class ResearchOutput(BaseModel):
    destination_overview: str       = Field(description="2-3 sentence overview of the destination")
    attractions:          list[str] = Field(description="Must-see landmarks and top attractions from web search")
    safety_notes:         str       = Field(description="Current safety warnings, crime info, health advisories")
    visa_info:            str       = Field(description="Visa/entry requirements for US/UK/EU citizens")
    best_time:            str       = Field(description="Best months to visit, seasonal weather, events to avoid/catch")
    local_customs:        list[str] = Field(description="Cultural etiquette, dress codes, tipping norms")
    transport_tips:       list[str] = Field(description="How to get around: metro, taxis, passes, apps")
    local_currency_context: str     = Field(description="Budget in local currency, e.g. '$3000 = ¥330,000 JPY'")
    destination_tier:     str       = Field(description="'budget' | 'mid' | 'luxury' for budget_allocator")

    @field_validator('attractions', 'local_customs', 'transport_tips', mode='before')
    @classmethod
    def parse_json_string_to_list(cls, v):
        if isinstance(v, str):
            import json
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except (json.JSONDecodeError, ValueError):
                pass
        return v

class ActivityDetail(BaseModel):
    name:        str
    time:        str              # e.g. "9:00 AM - 11:00 AM"
    duration_hrs: float
    cost_per_person_usd: float
    category:    str              # e.g. "culture", "food", "nature"
    description: str
    is_must_see: bool = False

class DayPlan(BaseModel):
    day:             int
    date:            str
    theme:           str                    # e.g. "Cultural Heritage Day"
    activities:      list[ActivityDetail]
    meals:           list[str]              # e.g. ["Breakfast at hotel", "Lunch: Tsukiji Market"]
    accommodation:   str
    daily_cost_per_person_usd: float
    notes:           Optional[str] = None   # e.g. "Book Louvre tickets online in advance"

class ItineraryOutput(BaseModel):
    summary:                  str            = Field(description="1-2 sentence trip overview")
    daily_plan:               list[DayPlan]
    total_estimated_cost_usd: float          = Field(description="Total for ALL travelers, not per-person")
    budget_breakdown:         dict           = Field(description="Per-person-per-day breakdown from budget_allocator")
    packing_list:             dict           = Field(description="Categorized packing list from packing_list_generator_tool")
    budget_status:            str            = Field(description="'within_budget' | 'over_budget' | 'under_budget'")
    recommendations:          list[str]      = Field(description="Pro tips: metro passes, free museum days, etc.")

class IntentValidation(BaseModel):
    understood_destination: str = Field(description="Destination parsed from user input")
    understood_dates: str = Field(description="Travel dates in natural language")
    understood_budget: str = Field(description="Budget with context (e.g. '$3000 for 2 people = $1500 per person')")
    understood_interests: list[str] = Field(description="List of interests/activities user wants")
    confirmation_message: str = Field(description="Natural language summary to show user for confirmation")
    needs_clarification: bool = Field(description="True if request is ambiguous/incomplete")
    clarification_questions: list[str] = Field(default_factory=list, description="Questions to ask if clarification needed")

class TravelPlanState(AgentState):
    session_id:       str
    request:          Optional[TravelRequest]
    intent_validation: Optional[dict]         # from IntentValidation.model_dump()
    intent_confirmed: bool = False            # User approved rephrased intent
    research_output:  Optional[dict]          # from ResearchOutput.model_dump()
    draft_itinerary:  Optional[dict]          # from ItineraryOutput.model_dump()
    hitl_feedback:    Optional[HITLFeedback]
    final_plan:       Optional[dict]
    status: Literal[
        "validating", "awaiting_intent_confirmation", "researching", "planning",
        "awaiting_review", "revising", "finalizing",
        "complete", "error",
    ]
    revision_count:  int
    error_message:   Optional[str]
