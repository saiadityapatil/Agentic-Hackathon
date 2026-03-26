import os
import json
from typing import Dict, List, Any, Optional
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)


def change_agent(state: "AgentState") -> Dict[str, Any]:
    """Analyze code changes and their potential impact on infrastructure.
    
    Evaluates:
    - Code change scope and complexity
    - Infrastructure impact of code changes
    - Risk assessment for deploying changes
    - Testing and validation requirements
    - Infrastructure capacity for new code patterns
    
    Args:
        state: Global agent state containing git_diff and code_analysis
        
    Returns:
        Dictionary with change_analysis and negotiation_history updates
    """
    git_diff = state.get('git_diff', '')
    code_analysis = state.get('code_analysis', {})
    cloud_stats = state.get('cloud_stats', {})
    performance_feedback = state.get('performance_feedback', [])
    history_context = "\n".join(state.get('negotiation_history', [])[-2:]) if state.get('negotiation_history') else ""
    
    # Format change context
    change_context = format_change_context(git_diff, code_analysis, cloud_stats)
    perf_text = json.dumps(performance_feedback, indent=2) if performance_feedback else "No performance feedback yet"
    
    prompt = f"""You are a DevOps/Infrastructure impact analysis specialist reviewing code changes.

Your role is to assess how code changes impact infrastructure requirements and stability.

You are NOT a code reviewer - focus on INFRASTRUCTURE IMPACT, not code quality.
You are NOT a performance engineer - rely on provided performance feedback.

{change_context}

Current Performance Feedback:
{perf_text}

{f'Previous context: {history_context}' if history_context else 'Perform initial change impact assessment.'}

Analyze the code changes and evaluate impact on:

1. Infrastructure Resource Impact
   - CPU/Memory requirement changes
   - Network bandwidth implications
   - Storage impact
   - Database connection/query pattern changes

2. Deployment Risk Assessment
   - Change complexity
   - Rollback difficulty
   - Blast radius of failure
   - Breaking changes to APIs or schema

3. Scaling Implications
   - Does code change support horizontal scaling?
   - New stateful patterns introduced?
   - Database migration needs?
   - Caching strategy changes?

4. Operational Impact
   - Monitoring requirements
   - Logging changes
   - Alerting rule updates needed
   - Operational procedures affected

IMPORTANT: Return ONLY a valid JSON object. Must have keys: impact_assessment, infrastructure_changes, risk_level, deployment_requirements, monitoring_needs, estimated_effort.

Example format:
{{
"impact_assessment": "Changes introduce async processing requiring queue infrastructure and worker scaling",
"infrastructure_changes": [
  "Add message queue service (SQS/RabbitMQ)",
  "Scale worker compute instances",
  "Adjust database connection pool size"
],
"risk_level": "Medium",
"deployment_requirements": [
  "Infrastructure changes must be deployed before code",
  "Database migration required for new event schema",
  "Worker auto-scaling rules need configuration"
],
"monitoring_needs": [
  "Queue depth monitoring",
  "Worker processing latency",
  "Dead letter queue metrics"
],
"estimated_effort": "High - requires coordinated infrastructure and code deployment"
}}"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()
    
    analysis = {}
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            analysis = json.loads(json_str)
        else:
            analysis = json.loads(response_text)
        
        if not isinstance(analysis, dict):
            analysis = {
                "impact_assessment": response_text,
                "infrastructure_changes": [],
                "risk_level": "Unknown",
                "deployment_requirements": [],
                "monitoring_needs": [],
                "estimated_effort": "Unknown"
            }
    except json.JSONDecodeError:
        analysis = {
            "impact_assessment": response_text,
            "infrastructure_changes": [],
            "risk_level": "Unknown",
            "deployment_requirements": [],
            "monitoring_needs": [],
            "estimated_effort": "Unknown"
        }
    
    change_msg = f"CODE CHANGE ANALYSIS (Turn {state.get('turn_count', 0)}):\n" + json.dumps(analysis, indent=2)
    
    return {
        "change_analysis": analysis,
        "negotiation_history": state.get('negotiation_history', []) + [change_msg]
    }


def format_change_context(git_diff: str, code_analysis: Dict[str, Any], cloud_stats: Dict[str, Any]) -> str:
    """Format code change context for analysis.
    
    Args:
        git_diff: Git diff showing code changes
        code_analysis: Structured code analysis from code_summarizer
        cloud_stats: Current cloud resource statistics
        
    Returns:
        Formatted string with change analysis context
    """
    context = "Code Change Impact Context:\n"
    context += "=" * 70 + "\n"
    
    # Git diff summary
    if git_diff:
        context += "\nGit Diff Summary:\n"
        lines = git_diff.split('\n')
        additions = sum(1 for line in lines if line.startswith('+'))
        deletions = sum(1 for line in lines if line.startswith('-'))
        context += f"  - Lines added: {additions}\n"
        context += f"  - Lines deleted: {deletions}\n"
        context += f"  - Files changed: {len(set(line.split(':')[0] for line in lines if '/' in line))}\n"
        
        # Show excerpt
        context += "\nChange Excerpt:\n"
        excerpt_lines = [l for l in lines if l.startswith(('+', '-'))][:20]
        for line in excerpt_lines:
            context += f"  {line[:80]}\n"
        if len(excerpt_lines) >= 20:
            context += "  ...\n"
    
    # Code analysis context
    if code_analysis:
        context += "\nExisting Code Architecture:\n"
        if 'application' in code_analysis:
            app = code_analysis['application']
            if 'framework' in app:
                context += f"  - Framework: {app['framework']}\n"
            if 'language' in app:
                context += f"  - Language: {app['language']}\n"
            if 'project_structure' in app:
                context += "  - Project Structure:\n"
                for key, val in app['project_structure'].items():
                    context += f"      - {key}: {val}\n"
    
    # Current infrastructure
    if cloud_stats:
        context += "\nCurrent Infrastructure Capacity:\n"
        for service, stats in cloud_stats.items():
            if isinstance(stats, dict):
                context += f"  - {service}:\n"
                for key, val in list(stats.items())[:3]:  # Limit to 3 key metrics
                    context += f"      - {key}: {val}\n"
    
    return context
