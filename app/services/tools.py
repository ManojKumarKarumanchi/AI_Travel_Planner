
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import requests
from langchain.tools import ToolRuntime, tool      # ToolRuntime is the doc pattern

@dataclass
class TravelContext:
    session_id: str
    user_id:    str = "anonymous"

# Tool 1 (Research) — Web search via Serper API

@tool
def web_search_tool(query: str) -> str:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return json.dumps({
            "error": "SERPER_API_KEY not configured",
            "message": "Get free API key at https://serper.dev (2,500 free queries)",
            "results": []
        })
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": 5},
            timeout=10,
        )
        resp.raise_for_status()
        results = [
            {"title": r.get("title"), "snippet": r.get("snippet"), "link": r.get("link")}
            for r in resp.json().get("organic", [])
        ]
        return json.dumps({"query": query, "results": results})
    except Exception as exc:
        return json.dumps({"error": str(exc), "results": []})

@tool
def currency_converter_tool(
    from_currency: str = "USD",
    to_currency: str = "EUR",
    amount: float = 1.0,
) -> str:
    api_key = os.getenv("EXCHANGERATE_API_KEY")
    base_url = "https://v6.exchangerate-api.com/v6"

    if not api_key:
        # Mock fallback for dev
        mock_rates = {"JPY": 110.0, "EUR": 0.85, "GBP": 0.73, "INR": 74.0, "AUD": 1.35}
        rate = mock_rates.get(to_currency.upper(), 1.0)
        return json.dumps({
            "from": from_currency.upper(),
            "to": to_currency.upper(),
            "amount": amount,
            "converted": round(amount * rate, 2),
            "rate": rate,
            "note": "EXCHANGERATE_API_KEY not set — mock rate used",
        })

    try:
        resp = requests.get(
            f"{base_url}/{api_key}/latest/{from_currency.upper()}",
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("result") != "success":
            raise ValueError(data.get("error-type", "Unknown API error"))

        rate = data["conversion_rates"].get(to_currency.upper())
        if not rate:
            raise ValueError(f"Currency {to_currency} not found in rates")

        return json.dumps({
            "from": from_currency.upper(),
            "to": to_currency.upper(),
            "amount": amount,
            "converted": round(amount * rate, 2),
            "rate": rate,
            "last_updated": data.get("time_last_update_utc", ""),
        })
    except Exception as exc:
        return json.dumps({"error": str(exc), "from": from_currency, "to": to_currency})

@tool
def packing_list_generator_tool(
    destination: str,
    duration_days: int,
    weather_type: str = "temperate",
    activities: str = "",
) -> str:
    # Base items always included
    essentials = {
        "Documents": ["Passport", "Travel insurance", "Tickets/confirmations", "Emergency contacts"],
        "Money": ["Credit/debit cards", "Local currency", "Emergency cash (USD/EUR)"],
        "Health": ["Prescription meds", "Basic first aid kit", "Hand sanitizer"],
        "Electronics": ["Phone + charger", "Power bank", "Universal adapter"],
    }

    # Weather-specific clothing
    clothing = {
        "tropical":  ["Lightweight shirts", "Shorts", "Swimwear", "Sun hat", "Sunglasses", "Sandals"],
        "temperate": ["T-shirts", "Long pants", "Light jacket", "Comfortable shoes", "Umbrella"],
        "cold":      ["Thermal layers", "Warm coat", "Gloves", "Scarf", "Winter boots", "Wool socks"],
        "desert":    ["Breathable long sleeves", "Wide-brim hat", "Sunglasses", "Light scarf", "Sunscreen"],
    }

    # Activity-specific gear
    activity_map = {
        "hiking": ["Hiking boots", "Daypack", "Water bottle", "Trail snacks"],
        "beach": ["Swimsuit", "Beach towel", "Reef-safe sunscreen", "Flip-flops"],
        "city": ["Walking shoes", "Day bag", "Camera", "Guidebook/map"],
        "skiing": ["Ski goggles", "Thermal gloves", "Base layers", "Ski socks"],
        "business": ["Formal attire", "Laptop", "Business cards", "Portfolio"],
    }

    packing = {**essentials}

    # Add weather-appropriate clothing
    weather_key = weather_type.lower().strip()
    packing["Clothing"] = clothing.get(weather_key, clothing["temperate"])

    # Add activity-specific items
    if activities:
        activity_list = [a.strip().lower() for a in activities.split(",")]
        gear = []
        for activity in activity_list:
            gear.extend(activity_map.get(activity, []))
        if gear:
            packing["Activity Gear"] = list(set(gear))  # deduplicate

    # Scale toiletries by duration
    toiletries = ["Toothbrush/paste", "Soap/shampoo", "Deodorant"]
    if duration_days > 7:
        toiletries.extend(["Laundry detergent packets", "Extra toiletries"])
    packing["Toiletries"] = toiletries

    return json.dumps({
        "destination": destination,
        "duration_days": duration_days,
        "weather_type": weather_type,
        "activities": activities,
        "packing_list": packing,
    })

@tool
def budget_allocator(
    total_budget_usd: int,
    num_days: int,
    num_travelers: int,
    destination_tier: str = "mid",
) -> str:
    tiers = {
        "budget":  {"accommodation": 0.30, "food": 0.25, "activities": 0.20,
                    "transport": 0.15, "contingency": 0.10},
        "mid":     {"accommodation": 0.35, "food": 0.25, "activities": 0.20,
                    "transport": 0.10, "contingency": 0.10},
        "luxury":  {"accommodation": 0.45, "food": 0.20, "activities": 0.20,
                    "transport": 0.08, "contingency": 0.07},
    }
    ratios  = tiers.get(destination_tier.lower(), tiers["mid"])
    per_pp  = total_budget_usd / max(num_travelers, 1)
    per_day = per_pp / max(num_days, 1)
    breakdown = {cat: round(per_day * pct, 2) for cat, pct in ratios.items()}
    return json.dumps({
        "total_usd": total_budget_usd, "travelers": num_travelers, "days": num_days,
        "per_person_per_day_usd": round(per_day, 2),
        "breakdown_per_person_per_day": breakdown,
        "destination_tier": destination_tier,
    })

# Exported tool lists — consumed by agents.py
# Assignment requires exactly 4 tools:

RESEARCH_TOOLS = [web_search_tool, currency_converter_tool]
PLANNER_TOOLS  = [budget_allocator, packing_list_generator_tool]
