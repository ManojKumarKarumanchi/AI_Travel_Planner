"""NVIDIA Client for AI Model Calls"""

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

client = NVIDIAEmbeddings(
    model="nvidia/nv-embed-v1",
    api_key="$NVIDIA_API_KEY",
    truncate="NONE",
)

embedding = client.embed_query("What is the capital of France?")
print(embedding)

from langchain_nvidia_ai_endpoints import ChatNVIDIA

client = ChatNVIDIA(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    api_key="$NVIDIA_API_KEY",
    temperature=1,
    top_p=0.95,
    max_tokens=16384,
    reasoning_budget=16384,
    chat_template_kwargs={"enable_thinking": True},
)

for chunk in client.stream([{"role": "user", "content": ""}]):

    if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
        print(chunk.additional_kwargs["reasoning_content"], end="")

    print(chunk.content, end="")

"""
Unified NVIDIA Chat Client with Dynamic Model Routing

Single abstraction for all NVIDIA model interactions.
Never instantiate ChatNVIDIA anywhere else in your codebase.
"""

import os
from functools import lru_cache
from typing import Literal, Optional, Any
from pydantic import BaseModel

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseChatModel

# MODEL REGISTRY - Change models here only

class NvidiaModels:
    """
    Central model registry for NVIDIA API Catalog models.
    
    All models support tool calling, structured output, and streaming.
    Update this class to change models across your entire application.
    """
    
    # Research & Analysis - DeepSeek R1 for reasoning
    RESEARCH = "deepseek-ai/deepseek-r1"
    
    # Planning & Orchestration - Nemotron Super for agentic tasks
    PLANNER = "nvidia/llama-3.3-nemotron-super-49b-v1"
    
    # Review & Quality - Llama 3.3 70B for evaluation
    REVIEW = "meta/llama-3.3-70b-instruct"
    
    # Summarization - Mixtral for concise outputs
    SUMMARIZER = "mistralai/mixtral-8x7b-instruct-v0.1"
    
    # Routing & Classification - Lightweight 8B model
    ROUTER = "meta/llama-3.1-8b-instruct"
    
    # Nemotron models for agentic AI (up to 1M context)
    NEMOTRON_SUPER = "nvidia/nemotron-3-super-120b-a12b"
    NEMOTRON = "nvidia/nemotron-3-8b"

# TASK TYPE DEFINITIONS

TaskType = Literal[
    "research",
    "planner", 
    "review",
    "summarizer",
    "router",
    "nemotron",
    "nemotron_super"
]

# LLM FACTORY

class NvidiaLLMFactory:
    """
    Factory for creating NVIDIA chat model instances.
    
    Features:
    - Centralized model configuration
    - LRU caching for efficiency
    - Task-specific model selection
    - Tool binding support
    - Structured output support
    - Streaming support
    
    Usage:
        # Basic usage
        llm = NvidiaLLMFactory.research()
        response = llm.invoke("Analyze this destination...")
        
        # With tools
        llm = NvidiaLLMFactory.with_tools("planner", [budget_tool, schedule_tool])
        
        # With structured output
        llm = NvidiaLLMFactory.with_structured_output("planner", ItinerarySchema)
        
        # Dynamic selection
        llm = NvidiaLLMFactory.get_llm("research")
    """
    
    @staticmethod
    @lru_cache(maxsize=20)
    def create(
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs: Any
    ) -> ChatNVIDIA:
        """
        Create a ChatNVIDIA instance with caching.
        
        Args:
            model: Model identifier from NvidiaModels
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional ChatNVIDIA parameters
            
        Returns:
            ChatNVIDIA instance
        """
        return ChatNVIDIA(
            model=model,
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    

    # TASK-SPECIFIC FACTORY METHODS

    
    @classmethod
    def research(cls, temperature: float = 0.1) -> ChatNVIDIA:
        """
        Research agent - DeepSeek R1 for deep reasoning.
        Low temperature for factual accuracy.
        """
        return cls.create(
            NvidiaModels.RESEARCH,
            temperature=temperature,
        )
    
    @classmethod
    def planner(cls, temperature: float = 0.4) -> ChatNVIDIA:
        """
        Planner agent - Nemotron Super for creative planning.
        Higher temperature for diverse suggestions.
        """
        return cls.create(
            NvidiaModels.PLANNER,
            temperature=temperature,
        )
    
    @classmethod
    def review(cls, temperature: float = 0.2) -> ChatNVIDIA:
        """
        Review agent - Llama 3.3 70B for quality evaluation.
        Moderate temperature for balanced assessment.
        """
        return cls.create(
            NvidiaModels.REVIEW,
            temperature=temperature,
        )
    
    @classmethod
    def summarizer(cls, temperature: float = 0.0) -> ChatNVIDIA:
        """
        Summarizer - Mixtral for concise, deterministic outputs.
        Zero temperature for consistent summaries.
        """
        return cls.create(
            NvidiaModels.SUMMARIZER,
            temperature=temperature,
        )
    
    @classmethod
    def router(cls, temperature: float = 0.0) -> ChatNVIDIA:
        """
        Router - Lightweight 8B model for fast classification.
        Zero temperature for deterministic routing.
        """
        return cls.create(
            NvidiaModels.ROUTER,
            temperature=temperature,
        )
    
    @classmethod
    def nemotron_super(cls, temperature: float = 0.3) -> ChatNVIDIA:
        """
        Nemotron Super - NVIDIA's flagship agentic model.
        Up to 1M context window for complex tasks.
        """
        return cls.create(
            NvidiaModels.NEMOTRON_SUPER,
            temperature=temperature,
        )
    

    # DYNAMIC MODEL SELECTION

    
    @classmethod
    def get_llm(cls, task: TaskType, **kwargs) -> ChatNVIDIA:
        """
        Dynamic model selection by task type.
        
        Args:
            task: Task type from TaskType literal
            **kwargs: Additional parameters (temperature, etc.)
            
        Returns:
            ChatNVIDIA instance configured for the task
            
        Example:
            llm = NvidiaLLMFactory.get_llm("research")
            llm = NvidiaLLMFactory.get_llm("planner", temperature=0.5)
        """
        task_mapping = {
            "research": cls.research,
            "planner": cls.planner,
            "review": cls.review,
            "summarizer": cls.summarizer,
            "router": cls.router,
            "nemotron": cls.nemotron_super,
            "nemotron_super": cls.nemotron_super,
        }
        
        factory_method = task_mapping.get(task)
        if not factory_method:
            raise ValueError(f"Unknown task type: {task}. Valid tasks: {list(task_mapping.keys())}")
        
        # Pass kwargs if provided, otherwise use defaults
        if kwargs:
            return factory_method(**kwargs)
        return factory_method()
    

    # TOOL BINDING

    
    @classmethod
    def with_tools(
        cls,
        task: TaskType,
        tools: list[BaseTool],
        **kwargs
    ) -> ChatNVIDIA:
        """
        Create LLM with tools bound for agent use.
        
        ChatNVIDIA supports native tool calling.
        
        Args:
            task: Task type
            tools: List of LangChain tools to bind
            **kwargs: Additional parameters
            
        Returns:
            ChatNVIDIA with tools bound
            
        Example:
            research_llm = NvidiaLLMFactory.with_tools(
                "research",
                [web_search_tool, weather_tool]
            )
        """
        llm = cls.get_llm(task, **kwargs)
        return llm.bind_tools(tools)
    

    # STRUCTURED OUTPUT

    
    @classmethod
    def with_structured_output(
        cls,
        task: TaskType,
        schema: type[BaseModel],
        **kwargs
    ) -> ChatNVIDIA:
        """
        Create LLM with structured output schema.
        
        ChatNVIDIA supports native structured output.
        
        Args:
            task: Task type
            schema: Pydantic model for output schema
            **kwargs: Additional parameters
            
        Returns:
            ChatNVIDIA configured for structured output
            
        Example:
            class TravelItinerary(BaseModel):
                days: list[DayPlan]
                total_cost: float
                
            planner_llm = NvidiaLLMFactory.with_structured_output(
                "planner",
                TravelItinerary
            )
        """
        llm = cls.get_llm(task, **kwargs)
        return llm.with_structured_output(schema)
    

    # SELF-HOSTED NIM SUPPORT

    
    @classmethod
    def self_hosted(
        cls,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        **kwargs
    ) -> ChatNVIDIA:
        """
        Connect to self-hosted NVIDIA NIM microservice.
        
        Args:
            base_url: URL of NIM deployment (e.g., "http://localhost:8000/v1")
            model: Model identifier
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            ChatNVIDIA connected to self-hosted NIM
            
        Example:
            llm = NvidiaLLMFactory.self_hosted(
                base_url="http://localhost:8000/v1",
                model="nvidia/nemotron-3-super-120b-a12b"
            )
        """
        return ChatNVIDIA(
            base_url=base_url,
            model=model,
            temperature=temperature,
            **kwargs
        )

# CONVENIENCE FUNCTIONS

def get_llm(task: TaskType, **kwargs) -> ChatNVIDIA:
    """
    Convenience function for dynamic model selection.
    
    Args:
        task: Task type
        **kwargs: Additional parameters
        
    Returns:
        ChatNVIDIA instance
    """
    return NvidiaLLMFactory.get_llm(task, **kwargs)

# USAGE EXAMPLES

if __name__ == "__main__":
    # Example 1: Basic usage
    research_llm = NvidiaLLMFactory.research()
    response = research_llm.invoke("What are the top attractions in Tokyo?")
    print(response.content)
    
    # Example 2: Dynamic selection
    llm = get_llm("planner")
    response = llm.invoke("Create a 3-day itinerary for Paris")
    print(response.content)
    
    # Example 3: With tools
    from langchain.tools import tool
    
    @tool
    def search_web(query: str) -> str:
        """Search the web for information."""
        return f"Results for: {query}"
    
    @tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: Sunny, 72°F"
    
    research_agent = NvidiaLLMFactory.with_tools(
        "research",
        [search_web, get_weather]
    )
    
    # Example 4: Structured output
    from pydantic import BaseModel, Field
    
    class DayPlan(BaseModel):
        day: int
        activities: list[str]
        estimated_cost: float
    
    class TravelItinerary(BaseModel):
        destination: str
        days: list[DayPlan]
        total_budget: float
    
    planner = NvidiaLLMFactory.with_structured_output("planner", TravelItinerary)
    itinerary = planner.invoke("Plan a 3-day trip to Tokyo with $500 budget")
    print(itinerary)
