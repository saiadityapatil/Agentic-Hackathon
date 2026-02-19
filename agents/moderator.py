import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from agents.finops import finops_agent

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)

def moderator_agent(finops_output, architecture_output, performance_output):
    prompt = f"""You are an Autonomous SRE Decision Engine.
    Your role is to synthesize and prioritize recommendations from:

    1) Architecture Agent
    2) Performance Agent
    3) FinOps Agent

    You are NOT allowed to generate new technical findings.
    You must ONLY reason over the provided agent outputs.

    Your job is to:

    - Detect conflicts
    - Evaluate risk
    - Prioritize actions
    - Produce a final implementation roadmap
    - Balance cost, performance, and reliability

    -------------------------------------------------

    ARCHITECTURE AGENT OUTPUT:
    {architecture_output}

    -------------------------------------------------

    PERFORMANCE AGENT OUTPUT:
    {performance_output}

    -------------------------------------------------

    FINOPS AGENT OUTPUT:
    {finops_output}

    -------------------------------------------------

    Your Responsibilities:

    1. Identify Conflicts
    - If FinOps suggests scaling down but Performance indicates high utilization → flag conflict.
    - If Architecture flags blocking DB calls and Performance shows DB saturation → elevate priority.
    - If cost savings increase reliability risk → deprioritize.

    2. Score Each Recommendation
    For each recommendation calculate:
    - Impact (High / Medium / Low)
    - Risk (Low / Medium / High)
    - Cost Benefit (High / Medium / Low)
    - Implementation Effort (Low / Medium / High)

    3. Rank Recommendations
    Produce a ranked list based on:
    - Production safety first
    - Performance stability second
    - Cost optimization third

    4. Generate Final Plan
    - Immediate Actions (Safe + High Impact)
    - Short-Term Improvements (Needs validation)
    - Long-Term Architecture Changes

    -------------------------------------------------

    Decision Rules:

    - Never prioritize cost reduction over system stability.
    - If system is near saturation (>80% CPU or DB utilization), scaling down is prohibited.
    - Architectural bottlenecks that cause cascading failure risk are highest priority.
    - Quick wins with low risk should be ranked above complex migrations.
    - If multiple agents agree on the same issue, elevate priority.
    - If agents conflict, explain why one recommendation is suppressed.

    -------------------------------------------------

    Return ONLY a valid JSON object.

    Structure:

    {{
    "conflicts_detected": [
        {{
        "description": string,
        "agents_involved": [string],
        "resolution_decision": string
        }}
    ],
    "ranked_recommendations": [
        {{
        "rank": integer,
        "recommendation": string,
        "impact": string,
        "risk": string,
        "cost_benefit": string,
        "implementation_effort": string,
        "rationale": string
        }}
    ],
    "implementation_plan": {{
        "immediate_actions": [string],
        "short_term": [string],
        "long_term": [string]
    }}
    }}
    No markdown.
    No commentary.
    No explanation outside JSON.
    """
    
    response = llm.invoke(prompt)
    response_text = response.content.strip()
    evaluation = json.loads(response_text)
    return evaluation

mock_azure_metrics = {
    "app_service": {
        "cpu_utilization": "25%",
        "memory_utilization": "40%",
        "instance_count": 2,
        "autoscaling_enabled": False
    },
    "sql_database": {
        "dtus": 100,
        "utilization": "60%",
        "cost_per_month": "$200"
    },
    "storage_account": {
        "total_storage_gb": 500,
        "active_storage_gb": 50,
        "cost_per_month": "$50"
    }
}

mock_azure_cost = {
    "app_service": {
        "cost_per_month": "$300"
    },
    "sql_database": {
        "cost_per_month": "$200"
    },
    "storage_account": {
        "cost_per_month": "$50"
    }
}

finops_output = finops_agent(mock_azure_metrics, mock_azure_cost)

print(json.dumps(finops_output, indent=2))


mock_architecture_output ={
  "architecture_style": "Monolithic Web Application",
  "cloud_topology": {
    "compute": "Azure App Service (2 instances, manual scaling)",
    "database": "Azure SQL Database (DTU-based tier)",
    "storage": "Azure Blob Storage",
    "caching_layer_present": False,
    "autoscaling_enabled": False
  },
  "design_observations": [
    "Application appears monolithic without separation of API and service layers",
    "Database access is synchronous which may limit scalability under high load",
    "No distributed caching layer detected",
    "Scaling strategy is manual rather than event-driven"
  ],
  "scalability_assessment": {
    "stateless_app": True,
    "horizontal_scaling_safe": True,
    "session_state_externalized": True,
    "bottleneck_component": "Azure SQL Database",
    "scalability_risk_level": "Medium"
  },
  "reliability_assessment": {
    "single_point_of_failure_detected": False,
    "autoscale_absent": True,
    "failover_strategy_detected": False,
    "resilience_score": 6
  },
  "performance_architecture_risks": [
    "Database-centric architecture may cause bottlenecks as traffic increases",
    "Synchronous database queries could block worker threads under high concurrency",
    "Lack of caching layer increases repeated database reads"
  ],
  "cost_architecture_risks": [
    "Manual scaling may cause overprovisioning during low traffic periods",
    "Storage tiering not aligned with access patterns"
  ],
  "production_readiness_score": 6.5,
  "architecture_recommendations": [
    {
      "recommendation": "Enable autoscaling for App Service",
      "impact": "High",
      "effort": "Low",
      "category": "Scalability"
    },
    {
      "recommendation": "Introduce Azure Cache for Redis for frequently accessed data",
      "impact": "High",
      "effort": "Medium",
      "category": "Performance"
    },
    {
      "recommendation": "Implement asynchronous database operations or connection pooling tuning",
      "impact": "Medium",
      "effort": "Medium",
      "category": "Concurrency"
    },
    {
      "recommendation": "Consider vCore-based SQL tier for predictable scaling",
      "impact": "Medium",
      "effort": "Medium",
      "category": "Database Architecture"
    }
  ]
}


mock_performance_output = [
  {
    "issue_detected": "Compute underutilization with static scaling configuration",
    "recommendation": "Enable autoscaling with minimum 1 instance and scale-out rules at 70% CPU",
    "impact": "Medium",
    "risk_level": "Low",
    "affected_services": ["App Service"],
    "confidence_level": "High"
  },
  {
    "issue_detected": "Database operating at moderate utilization without performance headroom for traffic spikes",
    "recommendation": "Configure SQL auto-scale or upgrade alert thresholds to prevent saturation during peak load",
    "impact": "Medium",
    "risk_level": "Medium",
    "affected_services": ["Azure SQL"],
    "confidence_level": "Medium"
  },
  {
    "issue_detected": "Lack of dynamic scaling strategy increases risk of inefficient resource usage during varying workloads",
    "recommendation": "Implement autoscale rules tied to CPU and request metrics",
    "impact": "Medium",
    "risk_level": "Low",
    "affected_services": ["App Service"],
    "confidence_level": "High"
  }
]


print("Moderator Agent Evaluation:")
moderator_evaluation = moderator_agent(finops_output, mock_architecture_output, mock_performance_output)
print(json.dumps(moderator_evaluation, indent=2))