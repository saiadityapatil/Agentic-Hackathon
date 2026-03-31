import json
import os
from langgraph.graph import StateGraph, END
from state import State
from agents.code_summarizer import code_summarizer_agent
from agents.moderator import moderator_agent
from agents.finops import finops_agent
from agents.architecture import architecture_agent
from agents.performance import performance_agent
from event_emitter import agent_emitter


def _load_json_if_present(path: str) -> dict:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def code_summarizer_node(state: State) -> dict:
    """
    Execute code summarizer agent with the repository URL.
    Stores the output in state for downstream agents.
    """
    print(f"🔍 Running Code Summarizer on: {state['repo_url']}")
    agent_emitter.emit_agent_started("code_summarizer")
    
    try:
        output = code_summarizer_agent(state, state['repo_url'])
        print("✅ Code Summarizer completed")
        agent_emitter.emit_agent_completed("code_summarizer", output)
        return {"code_summarizer_output": output}
    except Exception as e:
        print(f"❌ Code Summarizer failed: {str(e)}")
        agent_emitter.emit_agent_error("code_summarizer", str(e))
        return {"code_summarizer_output": {"error": str(e)}}


def architecture_node(state: State) -> dict:
    """
    Execute architecture agent.
    Takes code_summarizer_output as input for analysis.
    """
    print("🏗️ Running Architecture Agent")
    agent_emitter.emit_agent_started("architecture")
    
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        azure_metrics = _load_json_if_present(os.path.join(data_dir, "azure_metrics.json"))
        azure_cost = _load_json_if_present(os.path.join(data_dir, "azure_cost.json"))
        code_summary = state.get("code_summarizer_output", {}) or {}

        output = architecture_agent(code_summary, azure_metrics, azure_cost)
        print("✅ Architecture Agent completed")
        agent_emitter.emit_agent_completed("architecture", output)
        return {"architecture_output": output}
    except Exception as e:
        print(f"❌ Architecture Agent failed: {str(e)}")
        agent_emitter.emit_agent_error("architecture", str(e))
        return {"architecture_output": {"error": str(e)}}


def performance_node(state: State) -> dict:
    """
    Execute performance agent.
    """
    print("⚡ Running Performance Agent")
    agent_emitter.emit_agent_started("performance")
    
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        azure_metrics = _load_json_if_present(os.path.join(data_dir, "azure_metrics.json"))
        azure_cost = _load_json_if_present(os.path.join(data_dir, "azure_cost.json"))
        code_summary = state.get("code_summarizer_output", {}) or {}

        output = performance_agent(code_summary, azure_metrics, azure_cost)
        print("✅ Performance Agent completed")
        agent_emitter.emit_agent_completed("performance", output)
        return {"performance_output": output}
    except Exception as e:
        print(f"❌ Performance Agent failed: {str(e)}")
        agent_emitter.emit_agent_error("performance", str(e))
        return {"performance_output": {"error": str(e)}}


def finops_node(state: State) -> dict:
    """
    Execute finops agent.
    Analyzes cost optimization opportunities.
    """
    print("💰 Running FinOps Agent")
    agent_emitter.emit_agent_started("finops")
    
    try:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        azure_metrics = _load_json_if_present(os.path.join(data_dir, "azure_metrics.json"))
        azure_cost = _load_json_if_present(os.path.join(data_dir, "azure_cost.json"))

        output = finops_agent(azure_metrics, azure_cost)
        print("✅ FinOps Agent completed")
        agent_emitter.emit_agent_completed("finops", output)
        return {"finops_output": output}
    except Exception as e:
        print(f"❌ FinOps Agent failed: {str(e)}")
        agent_emitter.emit_agent_error("finops", str(e))
        return {"finops_output": {"error": str(e)}}


def moderator_node(state: State) -> dict:
    """
    Execute moderator agent.
    Takes outputs from architecture, performance, and finops agents.
    Synthesizes and prioritizes recommendations.
    """
    print("🎯 Running Moderator Agent")
    agent_emitter.emit_agent_started("moderator")
    
    try:
        architecture_output = state.get('architecture_output', {})
        performance_output = state.get('performance_output', {})
        finops_output = state.get('finops_output', {})
        
        output = moderator_agent(finops_output, architecture_output, performance_output)
        
        # Create final analysis summary
        final_analysis = {
            "status": "completed",
            "code_summarizer": state.get('code_summarizer_output', {}),
            "architecture": architecture_output,
            "performance": performance_output,
            "finops": finops_output,
            "moderator_synthesis": output
        }
        
        print("✅ Moderator Agent completed")
        agent_emitter.emit_agent_completed("moderator", output)
        return {
            "moderator_output": output,
            "final_analysis": json.dumps(final_analysis, indent=2)
        }
    except Exception as e:
        print(f"❌ Moderator Agent failed: {str(e)}")
        agent_emitter.emit_agent_error("moderator", str(e))
        return {
            "moderator_output": {"error": str(e)},
            "final_analysis": json.dumps({"status": "failed", "error": str(e)})
        }


def build_graph():
    """
    Build the LangChain StateGraph with the following execution flow:
    
    1. code_summarizer_node (runs first)
    2. architecture_node (sequential - uses code_summarizer output)
    3. performance_node & finops_node (run in parallel)
    4. moderator_node (runs last - consumes all three outputs)
    """
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("code_summarizer", code_summarizer_node)
    graph.add_node("architecture", architecture_node)
    graph.add_node("performance", performance_node)
    graph.add_node("finops", finops_node)
    graph.add_node("moderator", moderator_node)
    
    # Set entry point
    graph.set_entry_point("code_summarizer")
    
    # Define execution flow
    # After code_summarizer, architecture must run (sequential dependency)
    graph.add_edge("code_summarizer", "architecture")
    
    # After architecture, performance and finops run in parallel
    graph.add_edge("architecture", "performance")
    graph.add_edge("architecture", "finops")
    
    # Both performance and finops lead to moderator
    graph.add_edge("performance", "moderator")
    graph.add_edge("finops", "moderator")
    
    # Moderator is the end
    graph.set_finish_point("moderator")
    
    return graph.compile()


# Compile the graph
workflow = build_graph()
