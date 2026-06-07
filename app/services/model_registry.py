
from __future__ import annotations

import os
from typing import Literal, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv()

@dataclass
class ModelConfig:

    id: str
    name: str
    input_price: float  # USD per 1M tokens
    output_price: float  # USD per 1M tokens
    max_tokens: int
    context_window: int
    strengths: list[str]
    notes: str = ""

MODELS: dict[str, ModelConfig] = {
    "nemotron-super": ModelConfig(
        id="nvidia/nemotron-3-super-120b-a12b",
        name="Nemotron 3 Super 120B",
        input_price=0.50,
        output_price=2.00,
        max_tokens=4096,
        context_window=1_000_000,
        strengths=["tool_calling", "reasoning", "agentic_workflows"],
        notes="NVIDIA flagship model for agents - LangChain partnership",
    ),
    "llama-3.3-70b": ModelConfig(
        id="meta/llama-3.3-70b-instruct",
        name="Llama 3.3 70B Instruct",
        input_price=0.35,
        output_price=1.40,
        max_tokens=4096,
        context_window=128_000,
        strengths=["tool_calling", "fast_inference", "cost_effective"],
        notes="Meta's latest 70B - proven stable",
    ),
    "llama-3.1-70b": ModelConfig(
        id="meta/llama-3.1-70b-instruct",
        name="Llama 3.1 70B Instruct",
        input_price=0.35,
        output_price=1.40,
        max_tokens=4096,
        context_window=128_000,
        strengths=["tool_calling", "stable", "widely_tested"],
        notes="Fallback for 3.3 - battle-tested",
    ),
    "nemotron-49b": ModelConfig(
        id="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        name="Nemotron Super 49B v1.5",
        input_price=0.40,
        output_price=1.60,
        max_tokens=4096,
        context_window=128_000,
        strengths=["tool_calling", "balanced_cost"],
        notes="Mid-tier Nemotron - good balance",
    ),
    "llama-3.1-8b": ModelConfig(
        id="meta/llama-3.1-8b-instruct",
        name="Llama 3.1 8B Instruct",
        input_price=0.05,
        output_price=0.15,
        max_tokens=4096,
        context_window=128_000,
        strengths=["fast", "cheap", "basic_tool_calling"],
        notes="Emergency fallback - fast but less capable",
    ),
}

# Default fallback order (primary → emergency)
FALLBACK_ORDER = [
    "llama-3.3-70b",      # Primary - proven stable, good structured output
    "llama-3.1-70b",      # Fallback 1 - battle-tested
    "nemotron-49b",       # Fallback 2 - mid-tier balance
    "nemotron-super",     # Fallback 3 - powerful but generates malformed JSON
    "llama-3.1-8b",       # Emergency - fast/cheap
]

class ModelRegistry:

    def __init__(
        self,
        primary_model: Optional[str] = None,
        temperature: float = 1.0,
    ):
        self.temperature = temperature
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("NVIDIA_API_KEY not set")

        # Determine primary model
        env_model = os.getenv("NVIDIA_MODEL")
        if env_model:
            # Map full model ID to registry key
            self.primary_key = self._id_to_key(env_model)
        elif primary_model:
            self.primary_key = primary_model
        else:
            self.primary_key = "llama-3.3-70b"  # Default: best structured output reliability

        # Track current fallback index
        self._fallback_index = 0
        self._current_key = self.primary_key

    def _id_to_key(self, model_id: str) -> str:
        for key, config in MODELS.items():
            if config.id == model_id:
                return key
        # Unknown model - default to reliable primary
        return "llama-3.3-70b"

    def get_model(self, model_key: Optional[str] = None) -> ChatNVIDIA:
        key = model_key or self._current_key
        config = MODELS[key]

        return ChatNVIDIA(
            model=config.id,
            nvidia_api_key=self.api_key,
            temperature=self.temperature,
            top_p=1,
            max_tokens=config.max_tokens,
        )

    def get_fallback(self) -> tuple[ChatNVIDIA, str]:
        self._fallback_index += 1

        if self._fallback_index >= len(FALLBACK_ORDER):
            raise RuntimeError(
                f"All {len(FALLBACK_ORDER)} models failed. "
                "Check NVIDIA API status or API key."
            )

        self._current_key = FALLBACK_ORDER[self._fallback_index]
        config = MODELS[self._current_key]

        from app.utils.logger import get_logger
        logger = get_logger("model_registry")
        logger.warning(
            f"[FALLBACK] Switching to {config.name} ({self._current_key}) "
            f"- fallback {self._fallback_index}/{len(FALLBACK_ORDER)}"
        )

        return self.get_model(), self._current_key

    def reset(self):
        self._fallback_index = 0
        self._current_key = self.primary_key

    def get_config(self, model_key: Optional[str] = None) -> ModelConfig:
        key = model_key or self._current_key
        return MODELS[key]

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        model_key: Optional[str] = None,
    ) -> float:
        config = self.get_config(model_key)
        input_cost = (input_tokens / 1_000_000) * config.input_price
        output_cost = (output_tokens / 1_000_000) * config.output_price
        return input_cost + output_cost

    def list_models(self) -> dict[str, ModelConfig]:
        return MODELS

    def get_fallback_order(self) -> list[str]:
        return FALLBACK_ORDER[self._fallback_index:]

def get_reasoning_model(
    with_fallback: bool = False,
    temperature: float = 1.0,
) -> ChatNVIDIA | tuple[ChatNVIDIA, ModelRegistry]:
    registry = ModelRegistry(temperature=temperature)

    if with_fallback:
        return registry.get_model(), registry
    else:
        return registry.get_model()

def get_structured_model(temperature: float = 0.2) -> ChatNVIDIA:
    return ChatNVIDIA(
        model="meta/llama-3.1-8b-instruct",
        nvidia_api_key=os.getenv("NVIDIA_API_KEY"),
        temperature=temperature,
        top_p=1,
        max_tokens=2048,
    )
