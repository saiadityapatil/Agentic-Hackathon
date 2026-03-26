import json
from langgraph.graph import StateGraph, END
from state import State
from agents.code_summarizer import code_summarizer_agent
from agents.moderator import moderator_agent
from agents.finops import finops_agent
from event_emitter import agent_emitter


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
    Currently returns dummy output for demo/testing.
    """
    print("🏗️ Running Architecture Agent")
    agent_emitter.emit_agent_started("architecture")
    
    try:
        # For demo purposes, return dummy output
        # In production, this would process code_summarizer_output
        code_summary = state.get('code_summarizer_output', {})
        
        dummy_output = {
            "issues_detected": [
                "No autoscaling configured",
                "Database-centric architecture",
                "Single region deployment"
            ],
            "recommendations": [
                "Enable autoscaling on App Service",
                "Add Redis caching layer",
                "Consider multi-region setup"
            ],
            "framework": "FastAPI",
            "language": "Python",
            "project_structure": {
                "layered_architecture": True,
                "separate_service_layer": True,
                "repository_pattern": False,
                "monolithic": False,
                "microservice_ready": True,
                "circular_imports_detected": False
            },
            "api_design": {
                "route_count": 12,
                "uses_dependency_injection": True,
                "uses_pydantic_models": True,
                "validation_present": True,
                "pagination_supported": True
            }
        }
        
        print("✅ Architecture Agent completed (dummy output)")
        agent_emitter.emit_agent_completed("architecture", dummy_output)
        return {"architecture_output": dummy_output}
    except Exception as e:
        print(f"❌ Architecture Agent failed: {str(e)}")
        agent_emitter.emit_agent_error("architecture", str(e))
        return {"architecture_output": {"error": str(e)}}


def performance_node(state: State) -> dict:
    """
    Execute performance agent.
    Currently returns dummy output for demo/testing.
    """
    print("⚡ Running Performance Agent")
    agent_emitter.emit_agent_started("performance")
    
    try:
        # For demo purposes, return dummy output
        dummy_output = {
            "issues_detected": [
                "CPU utilization consistently low (25%)",
                "No dynamic scaling policies",
                "Cold start latency detected"
            ],
            "recommendations": [
                "Implement autoscaling rules",
                "Configure pre-warming",
                "Optimize connection pooling"
            ],
            "response_time_analysis": {
                "average_response_time_ms": 145,
                "p95_response_time_ms": 320,
                "p99_response_time_ms": 580
            },
            "throughput": {
                "requests_per_second": 450,
                "concurrent_connections_capacity": 1000,
                "current_load_percentage": 35
            },
            "bottlenecks": [
                "Database query on user endpoint taking 120ms",
                "Serialization overhead in list endpoints"
            ]
        }
        
        print("✅ Performance Agent completed (dummy output)")
        agent_emitter.emit_agent_completed("performance", dummy_output)
        return {"performance_output": dummy_output}
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
        # Get Azure metrics and costs from data
        import os
        from agents.finops import finops_agent, mock_azure_metrics, mock_azure_cost
        
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        azure_metrics_path = os.path.join(data_dir, 'azure_metrics.json')
        azure_cost_path = os.path.join(data_dir, 'azure_cost.json')
        
        # Default to mock data
        azure_metrics = mock_azure_metrics
        azure_cost = mock_azure_cost
        
        # Try to load from files, fallback to mock data if files are empty or invalid
        try:
            if os.path.exists(azure_metrics_path) and os.path.getsize(azure_metrics_path) > 0:
                with open(azure_metrics_path, 'r') as f:
                    loaded_metrics = json.load(f)
                    if loaded_metrics:  # Only use if not empty
                        azure_metrics = loaded_metrics
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Could not load azure_metrics.json, using mock data: {str(e)}")
        
        try:
            if os.path.exists(azure_cost_path) and os.path.getsize(azure_cost_path) > 0:
                with open(azure_cost_path, 'r') as f:
                    loaded_cost = json.load(f)
                    if loaded_cost:  # Only use if not empty
                        azure_cost = loaded_cost
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️  Could not load azure_cost.json, using mock data: {str(e)}")
        
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
