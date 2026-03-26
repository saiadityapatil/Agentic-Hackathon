"""Multi-agent orchestration graph for infrastructure analysis and decision making.

This module defines the LangGraph state machine that coordinates all agents:
1. Architecture Agent - Evaluates infrastructure design
2. Performance Agent - Assesses performance implications
3. FinOps Agent - Identifies cost optimizations
4. Change Agent - Analyzes code change impact
5. Moderator Agent - Synthesizes and prioritizes recommendations

Agents communicate through shared AgentState, with the moderator producing final decisions.
"""

import json
from typing import Dict, Any, List
from langgraph.graph import StateGraph, END
from state import AgentState
from agents.architecture import architecture_agent
from agents.performance import performance_agent
from agents.change import change_agent
from agents.finops import finops_agent
from agents.moderator import moderator_agent


def initialize_state(
    terraform_config: str = "",
    cloud_stats: Dict[str, Any] = None,
    git_diff: str = "",
    code_analysis: Dict[str, Any] = None
) -> AgentState:
    """Initialize the agent state with input data.
    
    Args:
        terraform_config: Terraform configuration content
        cloud_stats: Cloud service metrics and statistics
        git_diff: Git diff showing code changes
        code_analysis: Code structure analysis from code_summarizer
        
    Returns:
        Initialized AgentState dictionary
    """
    return {
        "terraform_config": terraform_config,
        "cloud_stats": cloud_stats or {},
        "code_analysis": code_analysis,
        "git_diff": git_diff,
        "architecture_feedback": None,
        "performance_feedback": None,
        "finops_proposals": None,
        "change_analysis": None,
        "conflicts_detected": None,
        "ranked_recommendations": None,
        "final_recommendation": None,
        "implementation_plan": None,
        "turn_count": 0,
        "negotiation_history": [],
        "status": "initialized"
    }


def create_analysis_graph() -> StateGraph:
    """Create the multi-agent orchestration graph.
    
    The graph follows this flow:
    1. Start -> Architecture Agent (evaluates design)
    2. Architecture -> Performance Agent (assesses performance)
    3. Architecture -> FinOps Agent (identifies cost optimizations)
    4. Architecture -> Change Agent (analyzes code impact)
    5. All agents -> Moderator Agent (synthesizes recommendations)
    6. Moderator -> End (generates final plan)
    
    Returns:
        Compiled LangGraph StateGraph
    """
    workflow = StateGraph(AgentState)
    
    # Define agent nodes
    workflow.add_node("architecture", architecture_node)
    workflow.add_node("performance", performance_node)
    workflow.add_node("finops", finops_node)
    workflow.add_node("change", change_node)
    workflow.add_node("moderator", moderator_node)
    
    # Define control flow
    workflow.set_entry_point("architecture")
    
    # Architecture triggers other agents in parallel
    workflow.add_edge("architecture", "performance")
    workflow.add_edge("architecture", "finops")
    workflow.add_edge("architecture", "change")
    
    # All gather at moderator
    workflow.add_edge("performance", "moderator")
    workflow.add_edge("finops", "moderator")
    workflow.add_edge("change", "moderator")
    
    # Moderator produces final output
    workflow.add_edge("moderator", END)
    
    return workflow.compile()


def architecture_node(state: AgentState) -> AgentState:
    """Node wrapper for architecture agent with error handling.
    
    Args:
        state: Current agent state
        
    Returns:
        Updated state with architecture_feedback
    """
    try:
        result = architecture_agent(state)
        state.update(result)
        state["status"] = "architecture_complete"
    except Exception as e:
        print(f"Architecture agent error: {e}")
        state["status"] = "architecture_error"
        state["architecture_feedback"] = [{"error": str(e)}]
    
    return state


def performance_node(state: AgentState) -> AgentState:
    """Node wrapper for performance agent with error handling.
    
    Args:
        state: Current agent state (includes architecture_feedback)
        
    Returns:
        Updated state with performance_feedback
    """
    try:
        result = performance_agent(state)
        state.update(result)
        state["status"] = "performance_complete"
    except Exception as e:
        print(f"Performance agent error: {e}")
        state["status"] = "performance_error"
        state["performance_feedback"] = [{"error": str(e)}]
    
    return state


def finops_node(state: AgentState) -> AgentState:
    """Node wrapper for FinOps agent with error handling.
    
    Args:
        state: Current agent state (includes architecture_feedback)
        
    Returns:
        Updated state with finops_proposals
    """
    try:
        result = finops_agent(state.get('cloud_stats', {}), {})
        state["finops_proposals"] = result if isinstance(result, list) else [result]
        
        finops_msg = f"FINOPS (analysis):\n" + json.dumps(result, indent=2)
        state["negotiation_history"] = state.get("negotiation_history", []) + [finops_msg]
        state["status"] = "finops_complete"
    except Exception as e:
        print(f"FinOps agent error: {e}")
        state["status"] = "finops_error"
        state["finops_proposals"] = [{"error": str(e)}]
    
    return state


def change_node(state: AgentState) -> AgentState:
    """Node wrapper for change agent with error handling.
    
    Args:
        state: Current agent state (includes architecture_feedback)
        
    Returns:
        Updated state with change_analysis
    """
    try:
        result = change_agent(state)
        state.update(result)
        state["status"] = "change_complete"
    except Exception as e:
        print(f"Change agent error: {e}")
        state["status"] = "change_error"
        state["change_analysis"] = {"error": str(e)}
    
    return state


def moderator_node(state: AgentState) -> AgentState:
    """Node wrapper for moderator agent that synthesizes all outputs.
    
    The moderator receives findings from all agents and produces:
    - Conflict detection
    - Prioritized recommendations
    - Implementation plan (immediate, short-term, long-term)
    
    Args:
        state: Current agent state (includes all agent outputs)
        
    Returns:
        Updated state with moderator decisions and final_recommendation
    """
    try:
        arch_output = json.dumps(state.get("architecture_feedback", []), indent=2)
        perf_output = json.dumps(state.get("performance_feedback", []), indent=2)
        finops_output = json.dumps(state.get("finops_proposals", []), indent=2)
        
        # Call moderator agent
        moderator_result = moderator_agent(finops_output, arch_output, perf_output)
        
        # Extract moderator decision
        if isinstance(moderator_result, dict):
            state["conflicts_detected"] = moderator_result.get("conflicts_detected", [])
            state["ranked_recommendations"] = moderator_result.get("ranked_recommendations", [])
            state["implementation_plan"] = moderator_result.get("implementation_plan", {})
            
            # Generate final recommendation
            state["final_recommendation"] = generate_final_recommendation(moderator_result)
        
        state["status"] = "moderator_complete"
    except Exception as e:
        print(f"Moderator agent error: {e}")
        state["status"] = "moderator_error"
        state["final_recommendation"] = f"Error in decision making: {str(e)}"
    
    return state


def generate_final_recommendation(moderator_result: Dict[str, Any]) -> str:
    """Generate human-readable final recommendation from moderator output.
    
    Args:
        moderator_result: Moderator agent decision dictionary
        
    Returns:
        Formatted final recommendation string
    """
    recommendation = "=== INFRASTRUCTURE DECISION ===\n\n"
    
    # Conflicts
    conflicts = moderator_result.get("conflicts_detected", [])
    if conflicts:
        recommendation += "CONFLICTS ADDRESSED:\n"
        for conflict in conflicts:
            recommendation += f"  - {conflict.get('description', 'Unknown conflict')}\n"
            recommendation += f"    Resolution: {conflict.get('resolution_decision', 'TBD')}\n"
        recommendation += "\n"
    
    # Top recommendations
    recommendations = moderator_result.get("ranked_recommendations", [])
    if recommendations:
        recommendation += "PRIORITIZED RECOMMENDATIONS:\n"
        for rec in recommendations[:5]:  # Top 5
            rank = rec.get("rank", "?")
            rec_text = rec.get("recommendation", "Unknown")
            impact = rec.get("impact", "Unknown")
            risk = rec.get("risk", "Unknown")
            recommendation += f"  {rank}. {rec_text}\n"
            recommendation += f"     Impact: {impact} | Risk: {risk}\n"
        recommendation += "\n"
    
    # Implementation plan
    plan = moderator_result.get("implementation_plan", {})
    if plan:
        recommendation += "IMPLEMENTATION ROADMAP:\n"
        
        immediate = plan.get("immediate_actions", [])
        if immediate:
            recommendation += "  Immediate Actions (Next 1-2 weeks):\n"
            for action in immediate[:3]:
                recommendation += f"    • {action}\n"
        
        short_term = plan.get("short_term", [])
        if short_term:
            recommendation += "  Short-term (1-2 months):\n"
            for action in short_term[:3]:
                recommendation += f"    • {action}\n"
        
        long_term = plan.get("long_term", [])
        if long_term:
            recommendation += "  Long-term (3+ months):\n"
            for action in long_term[:3]:
                recommendation += f"    • {action}\n"
    
    return recommendation
