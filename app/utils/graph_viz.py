"""
graph_viz.py — LangGraph workflow visualization utilities.

Functions:
  - generate_workflow_graph: Create base workflow diagram
  - generate_session_graph: Create session-specific graph with state
  - save_graph_metadata: Save session metadata JSON
  - get_mermaid_code: Export Mermaid diagram code
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

# Directories
FLOW_DIR = Path(__file__).parent.parent / "flow"
FLOW_DIR.mkdir(exist_ok=True)


def generate_workflow_graph(
    graph,
    output_filename: str = "workflow_base.png"
) -> Optional[str]:
    """
    Generate PNG visualization of base LangGraph workflow.

    Args:
        graph: Compiled LangGraph StateGraph
        output_filename: Filename for PNG (saved in app/flow/)

    Returns:
        Path to saved PNG file, or None if visualization unavailable
    """
    output_path = FLOW_DIR / output_filename

    try:
        # Get graph structure
        graph_drawable = graph.get_graph()

        # Try mermaid PNG generation (requires API or pyppeteer)
        try:
            from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle

            png_data = graph_drawable.draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
                curve_style=CurveStyle.LINEAR,
                background_color="white",
            )

            with open(output_path, "wb") as f:
                f.write(png_data)

            print(f"✅ Base workflow graph saved: {output_path}")
            return str(output_path)

        except Exception as mermaid_error:
            # Fallback: try ASCII representation
            print(f"⚠️  Mermaid PNG failed: {mermaid_error}")
            print("📝 Saving ASCII representation instead...")

            ascii_output = FLOW_DIR / "workflow_ascii.txt"
            ascii_repr = graph_drawable.print_ascii()

            with open(ascii_output, "w") as f:
                f.write(ascii_repr)

            print(f"✅ ASCII workflow saved: {ascii_output}")
            return str(ascii_output)

    except Exception as e:
        print(f"❌ Graph visualization failed: {e}")
        return None


def generate_session_graph(
    graph,
    session_id: str,
    state: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Generate session-specific graph visualization showing current execution state.

    Args:
        graph: Compiled LangGraph StateGraph
        session_id: Unique session identifier
        state: Current state dict (includes status, messages, etc.)

    Returns:
        Path to saved PNG file (app/flow/graph_{session_id}.png)
    """
    # Create safe filename from session ID
    safe_id = session_id.replace("-", "_")[:16]
    output_path = FLOW_DIR / f"graph_{safe_id}.png"

    try:
        # Get graph structure
        graph_drawable = graph.get_graph()

        # Extract current status for metadata
        current_status = state.get("status") if state else "unknown"
        revision_count = state.get("revision_count", 0) if state else 0

        # Generate PNG with session context
        try:
            from langchain_core.runnables.graph import MermaidDrawMethod, CurveStyle

            png_data = graph_drawable.draw_mermaid_png(
                draw_method=MermaidDrawMethod.API,
                curve_style=CurveStyle.LINEAR,
                background_color="#f9fafb",  # Light gray
            )

            with open(output_path, "wb") as f:
                f.write(png_data)

            # Save metadata alongside graph
            save_graph_metadata(session_id, state, str(output_path))

            print(f"✅ Session graph saved: {output_path}")
            print(f"   Status: {current_status} | Revisions: {revision_count}")

            return str(output_path)

        except Exception as mermaid_error:
            # Fallback: ASCII + metadata
            print(f"⚠️  Mermaid PNG failed: {mermaid_error}")

            ascii_output = FLOW_DIR / f"graph_{safe_id}.txt"
            ascii_repr = graph_drawable.print_ascii()

            with open(ascii_output, "w") as f:
                f.write(f"Session: {session_id}\n")
                f.write(f"Status: {current_status}\n")
                f.write(f"Revisions: {revision_count}\n")
                f.write("="*60 + "\n\n")
                f.write(ascii_repr)

            save_graph_metadata(session_id, state, str(ascii_output))

            print(f"✅ Session ASCII graph saved: {ascii_output}")
            return str(ascii_output)

    except Exception as e:
        print(f"❌ Session graph generation failed: {e}")
        return None


def save_graph_metadata(
    session_id: str,
    state: Optional[Dict[str, Any]],
    graph_path: str
) -> str:
    """
    Save session metadata JSON alongside graph visualization.

    Metadata includes:
      - Session ID
      - Current status
      - Timestamp
      - Graph image path
      - Revision count
      - Destination
      - Request details

    Args:
        session_id: Session identifier
        state: Current workflow state
        graph_path: Path to saved graph image

    Returns:
        Path to saved metadata JSON
    """
    safe_id = session_id.replace("-", "_")[:16]
    metadata_path = FLOW_DIR / f"meta_{safe_id}.json"

    metadata = {
        "session_id": session_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "graph_image": graph_path,
    }

    if state:
        metadata.update({
            "status": state.get("status", "unknown"),
            "revision_count": state.get("revision_count", 0),
            "intent_confirmed": state.get("intent_confirmed", False),
            "destination": state.get("request", {}).get("destination") if state.get("request") else None,
            "travelers": state.get("request", {}).get("travelers") if state.get("request") else None,
            "budget_usd": state.get("request", {}).get("budget_usd") if state.get("request") else None,
        })

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"📋 Metadata saved: {metadata_path}")
    return str(metadata_path)


def get_mermaid_code(graph) -> str:
    """
    Generate Mermaid diagram code (for Markdown/docs).

    Args:
        graph: Compiled LangGraph StateGraph

    Returns:
        Mermaid diagram as string
    """
    try:
        graph_drawable = graph.get_graph()
        return graph_drawable.draw_mermaid()
    except Exception as e:
        return f"```mermaid\n%% Mermaid diagram unavailable: {e}\n```"


def get_ascii_graph(graph) -> str:
    """
    Get ASCII representation of graph (fallback for no graphviz).

    Args:
        graph: Compiled LangGraph StateGraph

    Returns:
        ASCII art graph as string
    """
    try:
        graph_drawable = graph.get_graph()
        return graph_drawable.print_ascii()
    except Exception as e:
        return f"ASCII graph unavailable: {e}"


def cleanup_old_graphs(max_age_hours: int = 24):
    """
    Clean up graph images older than specified hours.

    Args:
        max_age_hours: Delete files older than this many hours
    """
    import time

    cutoff_time = time.time() - (max_age_hours * 3600)
    deleted_count = 0

    for file_path in FLOW_DIR.glob("graph_*.png"):
        if file_path.stat().st_mtime < cutoff_time:
            file_path.unlink()
            deleted_count += 1

            # Delete associated metadata
            meta_file = FLOW_DIR / f"meta_{file_path.stem.replace('graph_', '')}.json"
            if meta_file.exists():
                meta_file.unlink()

    if deleted_count > 0:
        print(f"🗑️  Cleaned up {deleted_count} old graph files")

    return deleted_count
