import os
import json
import re
from typing import Dict, List, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)


def performance_agent(state: "AgentState") -> Dict[str, Any]:
    """Evaluate performance implications of infrastructure design and proposals.
    
    Assesses:
    - Latency and throughput impact
    - Resource utilization patterns
    - Bottleneck identification
    - SLA compliance risks
    - Performance vs cost trade-offs
    
    Args:
        state: Global agent state containing cloud_stats and other feedback
        
    Returns:
        Dictionary with performance_feedback and negotiation_history updates
    """
    terraform_content = state.get('terraform_config', '')
    cloud_stats = state.get('cloud_stats', {})
    finops_proposals = state.get('finops_proposals', [])
    architecture_feedback = state.get('architecture_feedback', [])
    history_context = "\n".join(state.get('negotiation_history', [])[-2:]) if state.get('negotiation_history') else ""
    
    # Format performance context
    perf_context = format_performance_context(terraform_content, cloud_stats)
    proposals_text = json.dumps(finops_proposals, indent=2) if finops_proposals else "No FinOps proposals yet"
    arch_text = json.dumps(architecture_feedback, indent=2) if architecture_feedback else "No architecture feedback yet"
    
    prompt = f"""You are a Performance SRE evaluating infrastructure performance characteristics.

Your role is to assess performance impact and SLA/reliability risks.

You are NOT a cost optimizer - focus on PERFORMANCE and RELIABILITY, not pricing.
You are NOT an architect - evaluate given proposals, don't redesign.

{perf_context}

Previous Architecture Feedback:
{arch_text}

FinOps Proposals to Evaluate:
{proposals_text}

{f'Previous context: {history_context}' if history_context else 'Perform initial performance assessment.'}

Analyze performance characteristics and evaluate proposals for:

1. Performance Impact Assessment
   - Latency increase/decrease
   - Throughput implications
   - Resource saturation risks
   - Cold start or warm-up impacts

2. Utilization Patterns
   - Current vs safe utilization levels
   - Headroom for traffic spikes
   - Scaling readiness assessment
   - Bottleneck components

3. SLA & Reliability Risks
   - Availability percentage impact
   - Recovery time implications
   - Failover capability
   - Data consistency considerations

4. Resource Efficiency
   - Overprovisioning signals
   - Underutilization concerns
   - Right-sizing recommendations
   - Performance vs resource cost ratio

IMPORTANT: Return ONLY a valid JSON array. Start with [ and end with ]. Each object must have keys: issue, impact_level, affected_services, risk_assessment, mitigation_strategy, sla_implication.

Example format:
[
  {{"issue": "Database at 75% utilization during peak hours approaching saturation", "impact_level": "High", "affected_services": ["Database", "API"], "risk_assessment": "Potential timeout and connection pool exhaustion during traffic spikes", "mitigation_strategy": "Enable auto-scaling or upgrade instance tier, implement connection pooling", "sla_implication": "Risk of SLA breach (99.9% → 99.5%)"}},
  {{"issue": "Compute instances at 30% utilization with static scaling", "impact_level": "Medium", "affected_services": ["Compute"], "risk_assessment": "Over-provisioned during low traffic, vulnerable during sudden spikes", "mitigation_strategy": "Enable auto-scaling with CPU and request-based triggers", "sla_implication": "No impact if scaling works correctly"}}
]"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()
    
    feedback = []
    try:
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            feedback = json.loads(json_str)
        else:
            feedback = json.loads(response_text)
        
        if not isinstance(feedback, list):
            feedback = [{"issue": response_text, "impact_level": "Unknown", "affected_services": [], "risk_assessment": "N/A", "mitigation_strategy": "N/A", "sla_implication": "Unknown"}]
    except json.JSONDecodeError:
        feedback = [{"issue": response_text, "impact_level": "Unknown", "affected_services": [], "risk_assessment": "N/A", "mitigation_strategy": "N/A", "sla_implication": "Unknown"}]
    
    perf_msg = f"PERFORMANCE (Turn {state.get('turn_count', 0)}):\n" + json.dumps(feedback, indent=2)
    
    return {
        "performance_feedback": feedback,
        "negotiation_history": state.get('negotiation_history', []) + [perf_msg]
    }


def format_performance_context(terraform_content: str, cloud_stats: Dict[str, Any]) -> str:
    """Format performance-relevant infrastructure context.
    
    Args:
        terraform_content: Terraform configuration content
        cloud_stats: Cloud service metrics and statistics
        
    Returns:
        Formatted string with performance analysis context
    """
    analysis = "Performance & Utilization Context:\n"
    analysis += "=" * 70 + "\n"
    
    # Extract instance types and configurations
    instance_patterns = extract_instance_configurations(terraform_content)
    
    if instance_patterns:
        analysis += "\nCompute Configuration:\n"
        for category, items in instance_patterns.items():
            analysis += f"  - {category}: {', '.join(items) if items else 'None detected'}\n"
    
    # Analyze utilization metrics
    if cloud_stats:
        analysis += "\nCurrent Utilization Metrics:\n"
        for service, stats in cloud_stats.items():
            if isinstance(stats, dict):
                if 'cpu_utilization' in stats:
                    analysis += f"  - {service} CPU: {stats['cpu_utilization']}\n"
                if 'memory_utilization' in stats:
                    analysis += f"  - {service} Memory: {stats['memory_utilization']}\n"
                if 'utilization' in stats:
                    analysis += f"  - {service} Utilization: {stats['utilization']}\n"
                if 'instance_count' in stats:
                    analysis += f"  - {service} Instance Count: {stats['instance_count']}\n"
                if 'autoscaling_enabled' in stats:
                    analysis += f"  - {service} Auto-scaling: {stats['autoscaling_enabled']}\n"
    
    analysis += "\nAnalysis Summary:\n"
    analysis += "- Evaluate if utilization levels support expected traffic spikes\n"
    analysis += "- Assess scaling mechanisms and recovery time\n"
    analysis += "- Identify performance degradation risks\n"
    
    return analysis


def extract_instance_configurations(terraform_content: str) -> Dict[str, List[str]]:
    """Extract instance types and configurations from Terraform.
    
    Args:
        terraform_content: Terraform configuration content
        
    Returns:
        Dictionary of instance configurations
    """
    configs = {}
    
    # EC2 instance types
    ec2_pattern = r'instance_type\s*=\s*["\']([^"\']+)["\']'
    ec2_matches = re.findall(ec2_pattern, terraform_content)
    if ec2_matches:
        configs['EC2 Instances'] = list(set(ec2_matches))
    
    # RDS instance types
    rds_pattern = r'instance_class\s*=\s*["\']([^"\']+)["\']'
    rds_matches = re.findall(rds_pattern, terraform_content)
    if rds_matches:
        configs['RDS Instances'] = list(set(rds_matches))
    
    # AppService SKUs
    sku_pattern = r'sku\s*=\s*["\']([^"\']+)["\']'
    sku_matches = re.findall(sku_pattern, terraform_content)
    if sku_matches:
        configs['Service Tiers'] = list(set(sku_matches))
    
    return configs
