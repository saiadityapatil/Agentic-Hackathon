import json
import os
import time
from langgraph.graph import StateGraph, END
from state import State
from agents.code_summarizer import code_summarizer_agent
from agents.moderator import moderator_agent
from agents.finops import finops_agent
from agents.architecture import architecture_agent
from agents.performance import performance_agent
from event_emitter import agent_emitter
from data.azure_extractor import create_azure_extractor


def _is_rate_limit_error(error_msg: str) -> bool:
    """
    Check if an error message indicates a rate limit error.
    
    Args:
        error_msg: Error message string
        
    Returns:
        True if error is rate limit related
    """
    rate_limit_keywords = [
        "rate limit",
        "rate_limit",
        "ratelimit",
        "too many requests",
        "429",
        "quota exceeded",
        "throttled"
    ]
    error_lower = str(error_msg).lower()
    return any(keyword in error_lower for keyword in rate_limit_keywords)


def _retry_agent_on_rate_limit(agent_func, max_retries=3, retry_delay=10):
    """
    Decorator to retry agent execution on rate limit errors.
    
    Args:
        agent_func: Agent function to execute
        max_retries: Maximum number of retries (default: 3)
        retry_delay: Delay in seconds between retries (default: 10)
        
    Returns:
        Wrapper function that retries on rate limit
    """
    def wrapper(*args, **kwargs):
        retries = 0
        while retries <= max_retries:
            try:
                return agent_func(*args, **kwargs)
            except Exception as e:
                if _is_rate_limit_error(str(e)) and retries < max_retries:
                    retries += 1
                    print(f"⏳ Rate limit detected. Waiting {retry_delay}s before retry {retries}/{max_retries}...")
                    time.sleep(retry_delay)
                else:
                    # Not a rate limit error or max retries reached
                    raise
        return None
    return wrapper


def _load_azure_data(state: State) -> tuple:
    """
    Load Azure metrics and cost data either from extractor or static JSON files.
    Returns cached data if already loaded.
    
    Args:
        state: Current workflow state
        
    Returns:
        Tuple of (azure_metrics, azure_costs) dictionaries
    """
    # Check if we've already cached the data in state
    if state.get("azure_metrics") is not None and state.get("azure_costs") is not None:
        print("📦 Using cached Azure data from state")
        return state.get("azure_metrics"), state.get("azure_costs")
    
    # Check if Azure credentials are provided in state
    azure_creds = state.get("azure_credentials")
    resource_group = state.get("resource_group_name")
    
    if azure_creds and resource_group:
        # Use Azure extractor to fetch live data
        try:
            print(f"🔵 Fetching live Azure data from resource group: {resource_group}")
            extractor = create_azure_extractor(
                client_id=azure_creds.get("client_id"),
                client_secret=azure_creds.get("client_secret"),
                tenant_id=azure_creds.get("tenant_id"),
                subscription_id=azure_creds.get("subscription_id"),
                resource_group_name=resource_group
            )
            
            azure_metrics = extractor.extract_all_metrics()
            azure_costs = extractor.extract_all_costs()
            
            print("✅ Successfully fetched live Azure data")
            return azure_metrics, azure_costs
        except Exception as e:
            print(f"⚠️ Error fetching live Azure data: {str(e)}")
            print("📁 Falling back to static JSON files")
    
    # Fall back to static JSON files
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    azure_metrics = _load_json_if_present(os.path.join(data_dir, "azure_metrics.json"))
    azure_costs = _load_json_if_present(os.path.join(data_dir, "azure_cost.json"))
    
    return azure_metrics, azure_costs


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
    Also fetches and caches Azure metrics/costs if credentials provided.
    """
    print(f"🔍 Running Code Summarizer on: {state['repo_url']}")
    agent_emitter.emit_agent_started("code_summarizer")
    
    retries = 0
    max_retries = 3
    retry_delay = 10
    
    # Fetch Azure data once at the start (outside retry loop)
    azure_metrics, azure_costs = _load_azure_data(state)
    
    while retries <= max_retries:
        try:
            output = code_summarizer_agent(state, state['repo_url'])
            print("✅ Code Summarizer completed")
            agent_emitter.emit_agent_completed("code_summarizer", output)
            
            # Return both code summarizer output AND cached Azure data
            return {
                "code_summarizer_output": output,
                "azure_metrics": azure_metrics,
                "azure_costs": azure_costs
            }
        except Exception as e:
            error_msg = str(e)
            if _is_rate_limit_error(error_msg) and retries < max_retries:
                retries += 1
                print(f"⏳ Rate limit detected in Code Summarizer. Waiting {retry_delay}s before retry {retries}/{max_retries}...")
                time.sleep(retry_delay)
            else:
                # Not a rate limit error or max retries reached
                print(f"❌ Code Summarizer failed: {error_msg}")
                agent_emitter.emit_agent_error("code_summarizer", error_msg)
                return {"code_summarizer_output": {"error": error_msg}}
    
    # Should not reach here, but just in case
    return {"code_summarizer_output": {"error": "Max retries exceeded"}}


def architecture_node(state: State) -> dict:
    """
    Execute architecture agent with retry logic for rate limit errors.
    Takes code_summarizer_output as input for analysis.
    """
    print("🏗️ Running Architecture Agent")
    agent_emitter.emit_agent_started("architecture")
    
    retries = 0
    max_retries = 3
    retry_delay = 10
    
    while retries <= max_retries:
        try:
            azure_metrics, azure_costs = _load_azure_data(state)
            code_summary = state.get("code_summarizer_output", {}) or {}

            output = architecture_agent(code_summary, azure_metrics, azure_costs)
            print("✅ Architecture Agent completed")
            agent_emitter.emit_agent_completed("architecture", output)
            return {"architecture_output": output}
        except Exception as e:
            error_msg = str(e)
            if _is_rate_limit_error(error_msg) and retries < max_retries:
                retries += 1
                print(f"⏳ Rate limit detected in Architecture Agent. Waiting {retry_delay}s before retry {retries}/{max_retries}...")
                time.sleep(retry_delay)
            else:
                # Not a rate limit error or max retries reached
                print(f"❌ Architecture Agent failed: {error_msg}")
                agent_emitter.emit_agent_error("architecture", error_msg)
                return {"architecture_output": {"error": error_msg}}
    
    # Should not reach here, but just in case
    return {"architecture_output": {"error": "Max retries exceeded"}}


def performance_node(state: State) -> dict:
    """
    Execute performance agent with retry logic for rate limit errors.
    """
    print("⚡ Running Performance Agent")
    agent_emitter.emit_agent_started("performance")
    
    retries = 0
    max_retries = 3
    retry_delay = 10
    
    while retries <= max_retries:
        try:
            azure_metrics, azure_costs = _load_azure_data(state)
            code_summary = state.get("code_summarizer_output", {}) or {}

            output = performance_agent(code_summary, azure_metrics, azure_costs)
            print("✅ Performance Agent completed")
            agent_emitter.emit_agent_completed("performance", output)
            return {"performance_output": output}
        except Exception as e:
            error_msg = str(e)
            if _is_rate_limit_error(error_msg) and retries < max_retries:
                retries += 1
                print(f"⏳ Rate limit detected in Performance Agent. Waiting {retry_delay}s before retry {retries}/{max_retries}...")
                time.sleep(retry_delay)
            else:
                # Not a rate limit error or max retries reached
                print(f"❌ Performance Agent failed: {error_msg}")
                agent_emitter.emit_agent_error("performance", error_msg)
                return {"performance_output": {"error": error_msg}}
    
    # Should not reach here, but just in case
    return {"performance_output": {"error": "Max retries exceeded"}}


def finops_node(state: State) -> dict:
    """
    Execute finops agent with retry logic for rate limit errors.
    Analyzes cost optimization opportunities.
    """
    print("💰 Running FinOps Agent")
    agent_emitter.emit_agent_started("finops")
    
    retries = 0
    max_retries = 3
    retry_delay = 10
    
    while retries <= max_retries:
        try:
            azure_metrics, azure_costs = _load_azure_data(state)

            output = finops_agent(azure_metrics, azure_costs)
            print("✅ FinOps Agent completed")
            agent_emitter.emit_agent_completed("finops", output)
            return {"finops_output": output}
        except Exception as e:
            error_msg = str(e)
            if _is_rate_limit_error(error_msg) and retries < max_retries:
                retries += 1
                print(f"⏳ Rate limit detected in FinOps Agent. Waiting {retry_delay}s before retry {retries}/{max_retries}...")
                time.sleep(retry_delay)
            else:
                # Not a rate limit error or max retries reached
                print(f"❌ FinOps Agent failed: {error_msg}")
                agent_emitter.emit_agent_error("finops", error_msg)
                return {"finops_output": {"error": error_msg}}
    
    # Should not reach here, but just in case
    return {"finops_output": {"error": "Max retries exceeded"}}


def moderator_node(state: State) -> dict:
    """
    Execute moderator agent with retry logic for rate limit errors.
    Takes outputs from architecture, performance, and finops agents.
    Synthesizes and prioritizes recommendations.
    """
    print("🎯 Running Moderator Agent")
    agent_emitter.emit_agent_started("moderator")
    
    retries = 0
    max_retries = 3
    retry_delay = 10
    
    while retries <= max_retries:
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
            error_msg = str(e)
            if _is_rate_limit_error(error_msg) and retries < max_retries:
                retries += 1
                print(f"⏳ Rate limit detected in Moderator Agent. Waiting {retry_delay}s before retry {retries}/{max_retries}...")
                time.sleep(retry_delay)
            else:
                # Not a rate limit error or max retries reached
                print(f"❌ Moderator Agent failed: {error_msg}")
                agent_emitter.emit_agent_error("moderator", error_msg)
                return {
                    "moderator_output": {"error": error_msg},
                    "final_analysis": json.dumps({"status": "failed", "error": error_msg})
                }
    
    # Should not reach here, but just in case
    return {
        "moderator_output": {"error": "Max retries exceeded"},
        "final_analysis": json.dumps({"status": "failed", "error": "Max retries exceeded"})
    }


def build_graph():
    """
    Build the LangChain StateGraph with the following execution flow:
    
    1. code_summarizer_node (runs first)
    2. architecture_node (sequential - uses code_summarizer output)
    3. performance_node (sequential - runs after architecture)
    4. finops_node (sequential - runs after performance)
    5. moderator_node (runs last - consumes all three outputs)
    
    Note: All agents have retry logic for rate limit errors (3 retries, 10s delay)
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
    
    # Define execution flow - all sequential
    graph.add_edge("code_summarizer", "architecture")
    graph.add_edge("architecture", "performance")
    graph.add_edge("performance", "finops")
    graph.add_edge("finops", "moderator")
    
    # Moderator is the end
    graph.set_finish_point("moderator")
    
    return graph.compile()


# Compile the graph
workflow = build_graph()
