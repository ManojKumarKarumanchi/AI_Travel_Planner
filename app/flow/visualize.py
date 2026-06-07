"""
visualize.py — LangGraph workflow visualization.

Generates graph diagrams showing:
  - Node structure (validate, research, planner, HITL, finalize)
  - Edge routing (conditional branches)
  - Current execution state (per session)
  - Node status (pending, active, complete)

Usage:
  from app.flow.visualize import generate_graph_png, generate_session_graph

  # Generate base workflow diagram
  generate_graph_png(graph, output_path="workflow.png")

  # Generate session-specific diagram with current state
  generate_session_graph(graph, session_id, state_dict)
"""

import os
from pathlib import Path
from typing import Optional

try:
    from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
    from IPython.display import Image, display
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("[WARNING] Visualization libraries not available. Install: pip install pygraphviz pillow")


# Output directory for graph images
FLOW_DIR = Path(__file__).parent
FLOW_DIR.mkdir(exist_ok=True)


def generate_graph_png(
    graph,
    output_path: Optional[str] = None,
    title: str = "AI Travel Planner Workflow"
) -> str:
    """
    Generate PNG visualization of LangGraph workflow.

    Args:
        graph: Compiled LangGraph StateGraph
        output_path: Where to save PNG (default: app/flow/workflow.png)
        title: Title for the diagram

    Returns:
        Path to saved PNG file
    """
    if not VISUALIZATION_AVAILABLE:
        print("Visualization unavailable - install pygraphviz and pillow")
        return None

    if output_path is None:
        output_path = FLOW_DIR / "workflow.png"
    else:
        output_path = Path(output_path)

    try:
        # Get graph structure
        graph_drawable = graph.get_graph()

        # Generate PNG using mermaid
        png_data = graph_drawable.draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,  # or PYPPETEER for local rendering
            curve_style=CurveStyle.LINEAR,
            background_color="white",
        )

        # Save to file
        with open(output_path, "wb") as f:
            f.write(png_data)

        print(f"[OK] Graph visualization saved: {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"[FAIL] Graph visualization failed: {e}")
        print("Fallback: Use graph.get_graph().print_ascii() for text representation")
        return None


def generate_session_graph(
    graph,
    session_id: str,
    state: Optional[dict] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Generate session-specific graph visualization showing current state.

    Args:
        graph: Compiled LangGraph StateGraph
        session_id: Unique session identifier
        state: Current state dict (to highlight active nodes)
        output_path: Where to save (default: app/flow/{session_id}.png)

    Returns:
        Path to saved PNG file
    """
    if not VISUALIZATION_AVAILABLE:
        return None

    if output_path is None:
        # Save with session ID in filename
        safe_id = session_id.replace("-", "_")[:16]
        output_path = FLOW_DIR / f"graph_{safe_id}.png"
    else:
        output_path = Path(output_path)

    try:
        # Get graph structure
        graph_drawable = graph.get_graph()

        # If state provided, try to highlight current node
        current_status = state.get("status") if state else None

        # Generate PNG
        png_data = graph_drawable.draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
            curve_style=CurveStyle.LINEAR,
            background_color="#f9fafb",  # Light gray background
        )

        with open(output_path, "wb") as f:
            f.write(png_data)

        print(f"[OK] Session graph saved: {output_path} (status: {current_status})")
        return str(output_path)

    except Exception as e:
        print(f"[FAIL] Session graph failed: {e}")
        return None


def get_graph_ascii(graph) -> str:
    """
    Get ASCII representation of graph (fallback for environments without graphviz).

    Returns:
        String with ASCII art graph
    """
    try:
        graph_drawable = graph.get_graph()
        return graph_drawable.print_ascii()
    except Exception as e:
        return f"Graph ASCII representation unavailable: {e}"


def generate_mermaid_code(graph) -> str:
    """
    Generate Mermaid diagram code (can be rendered in Markdown/docs).

    Returns:
        Mermaid diagram as string
    """
    try:
        graph_drawable = graph.get_graph()
        return graph_drawable.draw_mermaid()
    except Exception as e:
        return f"```\nMermaid diagram unavailable: {e}\n```"


def visualize_workflow_structure():
    """
    Print detailed workflow structure to console.

    Shows:
      - All nodes
      - Edge connections
      - Conditional branches
      - HITL interrupts
    """
    print("\n" + "="*60)
    print("AI TRAVEL PLANNER - WORKFLOW STRUCTURE")
    print("="*60 + "\n")

    structure = """
    START
      ↓
    validate_node
      ├─[error]→ error_node → END
      └─[ok]──→ validate_intent_node (GPT-OSS structured output)
                  ↓
              intent_confirmation_node (HITL: confirm/edit)
                  ├─[confirm]→ run_research_node (Nemotron reasoning)
                  └─[edit]──→ back to validate_intent_node
                              ↓
                          run_planner_node (Nemotron reasoning)
                              ↓
                          hitl_node (HITL: approve/reject/modify)
                              ↓
                          revision_router (conditional)
                              ├─ "finalize"  → finalize_node → END
                              ├─ "planner"   → increment_revision → run_planner_node (loop)
                              └─ "research"  → increment_revision → run_research_node (loop)

    KEY:
      [VALIDATE] Blue nodes: Validation + Intent
      [AGENT] Green nodes: Agent execution (Research, Planner)
      [HITL] Yellow nodes: Human-in-the-loop (HITL interrupts)
      [ROUTER] Orange nodes: Control flow (routers, increment)
      [END] Red nodes: Terminal (finalize, error)

    TOOLS:
      Research Agent:
        - web_search_tool (Serper API)
        - currency_converter_tool (ExchangeRate API)

      Planner Agent:
        - budget_allocator (pure Python)
        - packing_list_generator_tool (pure Python)

    MODELS:
      - Intent Validation: openai/gpt-oss-20b (structured output)
      - Research Agent: nvidia/nemotron-3-nano-30b-a3b (reasoning)
      - Planner Agent: nvidia/nemotron-3-nano-30b-a3b (reasoning)
    """

    print(structure)
    print("="*60 + "\n")


def save_session_metadata(session_id: str, state: dict, graph_path: str):
    """
    Save session metadata alongside graph visualization.

    Creates JSON file with:
      - Session ID
      - Current status
      - Timestamp
      - Graph image path
      - Node history
    """
    import json
    from datetime import datetime

    metadata = {
        "session_id": session_id,
        "status": state.get("status", "unknown"),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "graph_image": graph_path,
        "revision_count": state.get("revision_count", 0),
        "destination": state.get("request", {}).get("destination"),
    }

    metadata_path = FLOW_DIR / f"meta_{session_id[:16]}.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"[META] Metadata saved: {metadata_path}")
    return str(metadata_path)


if __name__ == "__main__":
    # Demo: print workflow structure
    visualize_workflow_structure()

    print("\n[INFO] To generate graph visualization:")
    print("   from app.flow.visualize import generate_graph_png")
    print("   from workflow import graph")
    print("   generate_graph_png(graph, 'workflow.png')")
