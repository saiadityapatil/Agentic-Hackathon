import os
import json
import re
from typing import Dict, List, Any
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)


def architecture_agent(state: "AgentState") -> Dict[str, Any]:
    """Analyze infrastructure architecture and design patterns.
    
    Evaluates:
    - Cloud topology and service patterns
    - Scalability and reliability
    - Architectural bottlenecks
    - Design consistency
    
    Args:
        state: Global agent state containing terraform_config and cloud_stats
        
    Returns:
        Dictionary with architecture_feedback and negotiation_history updates
    """
    terraform_content = state.get('terraform_config', '')
    cloud_stats = state.get('cloud_stats', {})
    history_context = "\n".join(state.get('negotiation_history', [])[-2:]) if state.get('negotiation_history') else ""
    
    # Format infrastructure details
    infra_analysis = format_infrastructure_details(terraform_content, cloud_stats)
    
    prompt = f"""You are a Senior Cloud Architecture Reviewer analyzing infrastructure design.

Your role is to evaluate architectural patterns, scalability, reliability, and design quality.

You are NOT a performance engineer - focus on DESIGN and PATTERNS.
You are NOT a cost optimizer - focus on ARCHITECTURE, not pricing.

{infra_analysis}

{f'Previous context: {history_context}' if history_context else 'Perform initial architecture review.'}

Evaluate the infrastructure architecture and provide findings on:

1. Cloud Topology Assessment
   - Service patterns and interrelationships
   - Multi-region/zone strategy
   - Load balancing approach
   - Data persistence patterns

2. Scalability & Reliability
   - Horizontal/vertical scaling capability
   - High availability design
   - Failover mechanisms
   - Single points of failure

3. Architectural Bottlenecks
   - Synchronous vs asynchronous patterns
   - Database access patterns
   - Network latency considerations
   - State management approach

4. Design Quality
   - Separation of concerns
   - Dependency patterns
   - Infrastructure-as-code quality
   - Compliance with cloud best practices

IMPORTANT: Return ONLY a valid JSON array. Start with [ and end with ]. Each object must have keys: finding, category, severity, recommendation, affected_services.

Example format:
[
  {{"finding": "Monolithic architecture with synchronous database queries may limit horizontal scaling", "category": "Scalability", "severity": "Medium", "recommendation": "Implement asynchronous patterns and connection pooling", "affected_services": ["Compute", "Database"]}},
  {{"finding": "Single database instance without replica detected - single point of failure", "category": "Reliability", "severity": "High", "recommendation": "Configure read replicas and automated failover", "affected_services": ["Database"]}}
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
            feedback = [{"finding": response_text, "category": "Architecture", "severity": "Unknown", "recommendation": "N/A", "affected_services": []}]
    except json.JSONDecodeError:
        feedback = [{"finding": response_text, "category": "Architecture", "severity": "Unknown", "recommendation": "N/A", "affected_services": []}]
    
    arch_msg = f"ARCHITECTURE (Turn {state.get('turn_count', 0) + 1}):\n" + json.dumps(feedback, indent=2)
    
    return {
        "architecture_feedback": feedback,
        "negotiation_history": state.get('negotiation_history', []) + [arch_msg]
    }


def format_infrastructure_details(terraform_content: str, cloud_stats: Dict[str, Any]) -> str:
    """Format infrastructure details from Terraform and cloud statistics.
    
    Args:
        terraform_content: Terraform configuration file content
        cloud_stats: Cloud service statistics and metrics
        
    Returns:
        Formatted string with infrastructure analysis
    """
    resources = extract_resources_from_terraform(terraform_content)
    
    analysis = "Cloud Infrastructure Details:\n"
    analysis += "=" * 70 + "\n"
    
    # Add detected resources
    if resources:
        analysis += "\nDetected Cloud Resources:\n"
        for service, items in resources.items():
            analysis += f"  - {service}: {len(items)} resource(s)\n"
    
    # Add cloud statistics
    if cloud_stats:
        analysis += "\nCloud Service Statistics & Metrics:\n"
        for service, stats in cloud_stats.items():
            analysis += f"  - {service}:\n"
            if isinstance(stats, dict):
                for key, value in stats.items():
                    analysis += f"      {key}: {value}\n"
            else:
                analysis += f"      {stats}\n"
    
    # Add configuration excerpt
    if terraform_content:
        analysis += "\nConfiguration Excerpt:\n"
        excerpt = terraform_content[:700]
        analysis += excerpt + ("..." if len(terraform_content) > 700 else "")
    
    return analysis


def extract_resources_from_terraform(terraform_content: str) -> Dict[str, List[str]]:
    """Extract cloud resource types from Terraform configuration.
    
    Args:
        terraform_content: Terraform configuration content
        
    Returns:
        Dictionary mapping service names to list of resource identifiers
    """
    resources = {}
    
    # AWS pattern: resource "aws_service_name" "name" { ... }
    aws_pattern = r'resource\s+"(aws_[^"]+)"\s+"[^"]+"\s+\{'
    aws_matches = re.findall(aws_pattern, terraform_content)
    
    for match in aws_matches:
        service = match.split('_', 1)[1].upper().replace('_', ' ')
        if service not in resources:
            resources[service] = []
        resources[service].append(match)
    
    # Azure pattern: resource "azurerm_service_name" "name" { ... }
    azure_pattern = r'resource\s+"(azurerm_[^"]+)"\s+"[^"]+"\s+\{'
    azure_matches = re.findall(azure_pattern, terraform_content)
    
    for match in azure_matches:
        service = match.split('_', 1)[1].upper().replace('_', ' ')
        if service not in resources:
            resources[service] = []
        resources[service].append(match)
    
    return resources
