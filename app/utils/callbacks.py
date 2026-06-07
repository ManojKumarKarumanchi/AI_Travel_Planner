
import time
from typing import Any, Dict, List, Optional
from datetime import datetime

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.utils.logger import get_logger

logger = get_logger("callbacks")

NVIDIA_PRICING = {
    # Nemotron models (NVIDIA official)
    "nvidia/nemotron-3-super-120b-a12b": {"input": 0.50, "output": 2.00},  # Flagship agentic model
    "nvidia/llama-3.3-nemotron-super-49b-v1.5": {"input": 0.40, "output": 1.60},
    "nvidia/llama-3.1-nemotron-ultra-253b-v1": {"input": 1.00, "output": 4.00},
    "nvidia/nemotron-3-nano-30b-a3b": {"input": 0.05, "output": 0.20},  # UNSTABLE - 500 errors on 3rd call
    # Meta Llama models
    "meta/llama-3.1-8b-instruct": {"input": 0.05, "output": 0.15},
    "meta/llama-3.1-70b-instruct": {"input": 0.35, "output": 1.40},
    "meta/llama-3.3-70b-instruct": {"input": 0.35, "output": 1.40},
}

class TimingCostCallback(BaseCallbackHandler):

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._llm_start_times: Dict[str, float] = {}
        self._tool_start_times: Dict[str, float] = {}

    # LLM callbacks

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._llm_start_times[str(run_id)] = time.perf_counter()
        model = kwargs.get("invocation_params", {}).get("model", "unknown")
        logger.info(f"[LLM_START] session={self.session_id[:8]} | model={model} | run_id={str(run_id)[:8]}")

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)

        # Calculate duration
        start_time = self._llm_start_times.pop(run_id_str, None)
        duration_ms = int((time.perf_counter() - start_time) * 1000) if start_time else 0

        # Extract token usage
        llm_output = response.llm_output or {}
        token_usage = llm_output.get("token_usage", {})
        input_tokens = token_usage.get("prompt_tokens", 0) or token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("completion_tokens", 0) or token_usage.get("output_tokens", 0)
        total_tokens = token_usage.get("total_tokens", input_tokens + output_tokens)

        # Extract model name
        model = llm_output.get("model_name", "unknown")

        # Calculate cost
        cost = self._estimate_cost(model, input_tokens, output_tokens)

        # Structured log
        logger.info(
            f"[LLM_END] session={self.session_id[:8]} | "
            f"model={model} | "
            f"duration={duration_ms}ms | "
            f"tokens={total_tokens} (in={input_tokens}, out={output_tokens}) | "
            f"cost=${cost:.4f} | "
            f"run_id={run_id_str[:8]}"
        )

    def on_llm_error(
        self,
        error: Exception,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        self._llm_start_times.pop(run_id_str, None)
        logger.error(f"[LLM_ERROR] session={self.session_id[:8]} | error={str(error)[:100]} | run_id={run_id_str[:8]}")

    # Tool callbacks

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        self._tool_start_times[str(run_id)] = time.perf_counter()
        tool_name = serialized.get("name", "unknown")

        # Truncate long inputs
        input_preview = input_str[:100] + "..." if len(input_str) > 100 else input_str

        logger.info(
            f"[TOOL_START] session={self.session_id[:8]} | "
            f"tool={tool_name} | "
            f"input={input_preview} | "
            f"run_id={str(run_id)[:8]}"
        )

    def on_tool_end(
        self,
        output: Any,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)

        # Calculate duration
        start_time = self._tool_start_times.pop(run_id_str, None)
        duration_ms = int((time.perf_counter() - start_time) * 1000) if start_time else 0

        output_str = str(output)

        # Truncate long outputs
        output_preview = output_str[:100] + "..." if len(output_str) > 100 else output_str

        logger.info(
            f"[TOOL_END] session={self.session_id[:8]} | "
            f"duration={duration_ms}ms | "
            f"output={output_preview} | "
            f"run_id={run_id_str[:8]}"
        )

    def on_tool_error(
        self,
        error: Exception,
        *,
        run_id: Any,
        **kwargs: Any,
    ) -> None:
        run_id_str = str(run_id)
        self._tool_start_times.pop(run_id_str, None)
        logger.error(f"[TOOL_ERROR] session={self.session_id[:8]} | error={str(error)[:100]} | run_id={run_id_str[:8]}")

    # Cost estimation

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        pricing = NVIDIA_PRICING.get(model)
        if not pricing:
            return 0.0

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]

        return input_cost + output_cost

class SessionMetricsTracker:

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.llm_calls = 0
        self.tool_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        self.total_duration_ms = 0
        self.start_time = time.perf_counter()

    def record_llm_call(self, duration_ms: int, input_tokens: int, output_tokens: int, cost: float):
        self.llm_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost += cost
        self.total_duration_ms += duration_ms

    def record_tool_call(self, duration_ms: int):
        self.tool_calls += 1
        self.total_duration_ms += duration_ms

    def get_summary(self) -> Dict[str, Any]:
        elapsed_ms = int((time.perf_counter() - self.start_time) * 1000)

        return {
            "session_id": self.session_id,
            "elapsed_time_ms": elapsed_ms,
            "total_duration_ms": self.total_duration_ms,  # Sum of all call durations
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost, 4),
            "avg_llm_latency_ms": int(self.total_duration_ms / self.llm_calls) if self.llm_calls > 0 else 0,
        }

    def log_summary(self):
        summary = self.get_summary()
        logger.info(
            f"[SESSION_SUMMARY] {self.session_id[:8]} | "
            f"elapsed={summary['elapsed_time_ms']}ms | "
            f"llm_calls={summary['llm_calls']} | "
            f"tool_calls={summary['tool_calls']} | "
            f"tokens={summary['total_tokens']} | "
            f"cost=${summary['total_cost_usd']}"
        )
