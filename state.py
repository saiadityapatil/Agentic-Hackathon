from typing import TypedDict, List, Optional, Dict, Any


class AgentState(TypedDict):
    """Shared state across all agents in the multi-agent system.
    
    This state is passed through the graph and updated by each agent node.
    Agents communicate their findings through this shared state.
    """
    # Infrastructure & Code data
    terraform_config: str  # Terraform configuration content
    cloud_stats: Dict[str, Any]  # Cloud metrics (AWS/Azure)
    code_analysis: Optional[Dict[str, Any]]  # Code structure analysis from code_summarizer
    git_diff: Optional[str]  # Git diff/changes for code impact analysis
    
    # Agent outputs (findings)
    architecture_feedback: Optional[List[Dict]]  # Architecture agent findings
    performance_feedback: Optional[List[Dict]]  # Performance agent findings
    finops_proposals: Optional[List[Dict]]  # FinOps cost optimization proposals
    change_analysis: Optional[Dict[str, Any]]  # Code change impact analysis
    
    # Moderation & Decision Making
    conflicts_detected: Optional[List[Dict]]  # Conflicts identified by moderator
    ranked_recommendations: Optional[List[Dict]]  # Prioritized recommendations
    final_recommendation: Optional[str]  # Final decision from moderator
    implementation_plan: Optional[Dict[str, List[str]]]  # Actionable plan
    
    # Metadata
    turn_count: int  # Track number of agent turns
    negotiation_history: List[str]  # Historical context of agent discussions
    status: str  # Current workflow status (e.g., "running", "completed", "error")
