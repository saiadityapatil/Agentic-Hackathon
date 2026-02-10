import os
import json
import re
from typing import TypedDict, List, Optional, Dict, Any
from pathlib import Path
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file if present

# 1. Setup Groq API Key
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")  # Replace with your actual API key or set in .env
# 2. Define LLM
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)

class AgentState(TypedDict):
    terraform_config: str
    cloud_stats: Dict[str, Any]
    finops_proposals: List[Dict]
    performance_feedback: List[Dict]
    final_recommendation: Optional[str]
    turn_count: int
    negotiation_history: List[str]

# --- Utility Functions ---

def parse_terraform_file(file_path: str) -> str:
    """Read and parse Terraform configuration file"""
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Terraform file not found at {file_path}")
        return ""

def extract_aws_resources_from_terraform(terraform_content: str) -> Dict[str, List[str]]:
    """Extract AWS resources from Terraform configuration"""
    resources = {}
    
    # Pattern to match resource blocks: resource "aws_service_name" "name" { ... }
    resource_pattern = r'resource\s+"(aws_[^"]+)"\s+"[^"]+"\s+\{'
    matches = re.findall(resource_pattern, terraform_content)
    
    for match in matches:
        # Extract service name and count
        service = match.split('_', 1)[1].upper().replace('_', ' ')
        if service not in resources:
            resources[service] = []
        resources[service].append(match)
    
    return resources

def format_infrastructure_analysis(terraform_content: str, cloud_stats: Dict) -> str:
    """Format infrastructure analysis from Terraform and cloud stats"""
    resources = extract_aws_resources_from_terraform(terraform_content)
    
    analysis = "AWS Infrastructure Analysis:\n"
    analysis += "=" * 60 + "\n"
    
    # Add Terraform resources
    if resources:
        analysis += "\nTerraform Resources Detected:\n"
        for service, items in resources.items():
            analysis += f"  - {service}: {len(items)} resource(s)\n"
    
    # Add cloud statistics
    if cloud_stats:
        analysis += "\nCloud Service Statistics:\n"
        for service, stats in cloud_stats.items():
            analysis += f"  - {service}:\n"
            if isinstance(stats, dict):
                for key, value in stats.items():
                    analysis += f"      {key}: {value}\n"
            else:
                analysis += f"      {stats}\n"
    
    analysis += "\nTerraform Configuration (excerpt):\n"
    excerpt = terraform_content[:600]
    analysis += excerpt + ("..." if len(terraform_content) > 600 else "")
    
    return analysis

# --- Agent Nodes ---

def finops_agent(state: AgentState):
    """Generate cost-cutting proposals with estimated savings"""
    infra_analysis = format_infrastructure_analysis(state['terraform_config'], state['cloud_stats'])
    history_context = "\n".join(state['negotiation_history'][-2:]) if state['negotiation_history'] else ""
    
    prompt = f"""You are a FinOps Cloud Economist analyzing infrastructure costs across multiple AWS services.

{infra_analysis}

{f'Previous feedback: {history_context}' if history_context else 'Generate initial proposals.'}

Analyze the infrastructure and provide 3 specific cost-cutting proposals. For each, consider:
1. Service-specific optimization (EC2, RDS, S3, Lambda, DynamoDB, etc.)
2. Estimated monthly savings as percentage and dollar amount
3. Implementation complexity (Low/Medium/High)
4. Multi-service impact

IMPORTANT: Return ONLY a valid JSON array. Start with [ and end with ]. Each object must have keys: description, estimated_savings, complexity, affected_services.

Example format:
[
  {{"description": "Migrate underutilized EC2 instances to Lambda for API processing", "estimated_savings": "25% (~$500/month)", "complexity": "High", "affected_services": ["EC2", "Lambda"]}},
  {{"description": "Enable S3 Intelligent-Tiering and compress DynamoDB data", "estimated_savings": "15% (~$150/month)", "complexity": "Low", "affected_services": ["S3", "DynamoDB"]}}
]"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()
    
    proposals = []
    try:
        start_idx = response_text.find('[')
        end_idx = response_text.rfind(']') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            proposals = json.loads(json_str)
        else:
            proposals = json.loads(response_text)
        
        if not isinstance(proposals, list):
            proposals = [{"description": response_text, "estimated_savings": "TBD", "complexity": "Unknown", "affected_services": []}]
    except json.JSONDecodeError:
        proposals = [{"description": response_text, "estimated_savings": "TBD", "complexity": "Unknown", "affected_services": []}]
    
    finops_msg = f"FINOPS (Turn {state['turn_count'] + 1}):\n" + json.dumps(proposals, indent=2)
    
    return {
        "finops_proposals": proposals,
        "turn_count": state['turn_count'] + 1,
        "negotiation_history": state['negotiation_history'] + [finops_msg]
    }

def performance_agent(state: AgentState):
    """Evaluate performance risks of each proposal across services"""
    infra_analysis = format_infrastructure_analysis(state['terraform_config'], state['cloud_stats'])
    proposals_text = json.dumps(state['finops_proposals'], indent=2)
    
    prompt = f"""You are a Performance SRE evaluating infrastructure changes across AWS services.

{infra_analysis}

FinOps proposals:
{proposals_text}

For each proposal, assess:
1. Performance risk level (Low/Medium/High)
2. Service-specific latency/availability impact
3. Mitigation strategy
4. Impact on SLA compliance

IMPORTANT: Return ONLY a valid JSON array with one object per proposal. Start with [ and end with ]. Each object must have keys: proposal_index, risk_level, impact, mitigation, sla_impact.

Example format:
[
  {{"proposal_index": 0, "risk_level": "Medium", "impact": "Potential 50ms latency increase during Lambda cold starts", "mitigation": "Use provisioned concurrency and connection pooling", "sla_impact": "99.9% to 99.5%"}},
  {{"proposal_index": 1, "risk_level": "Low", "impact": "No significant impact on performance", "mitigation": "Enable S3 versioning and DynamoDB auto-scaling", "sla_impact": "No change, 99.9%"}}
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
            feedback = [{"risk_level": "Unknown", "impact": response_text, "mitigation": "N/A", "sla_impact": "Unknown"}]
    except json.JSONDecodeError:
        feedback = [{"risk_level": "Unknown", "impact": response_text, "mitigation": "N/A", "sla_impact": "Unknown"}]
    
    perf_msg = f"PERFORMANCE (Turn {state['turn_count']}):\n" + json.dumps(feedback, indent=2)
    
    return {
        "performance_feedback": feedback,
        "negotiation_history": state['negotiation_history'] + [perf_msg]
    }

def moderator_agent(state: AgentState):
    """Compare trade-offs and recommend optimal architecture"""
    infra_analysis = format_infrastructure_analysis(state['terraform_config'], state['cloud_stats'])
    proposals_text = json.dumps(state['finops_proposals'], indent=2)
    feedback_text = json.dumps(state['performance_feedback'], indent=2)
    
    prompt = f"""You are the Lead Cloud Architect making final infrastructure decisions.

{infra_analysis}

FinOps Proposals:
{proposals_text}

Performance Risk Assessment:
{feedback_text}

Your task:
1. Analyze each proposal's cost vs risk trade-off
2. Consider multi-service impact and dependencies
3. Rank the top 3 options
4. Recommend the #1 best option with implementation roadmap
5. Provide Terraform modification suggestions

IMPORTANT: Return ONLY a valid JSON object. Start with {{ and end with }}. Keys: ranking, top_recommendation, implementation_steps, terraform_changes, rationale.

Example format:
{{
  "ranking": [
    {{"index": 0, "description": "Migrate to Lambda", "savings": "25%", "risk": "Medium", "score": 8, "affected_services": ["EC2", "Lambda"]}}
  ],
  "top_recommendation": "Implement multi-service cost optimization prioritizing Lambda migration...",
  "implementation_steps": ["Step 1: Audit EC2 workloads", "Step 2: Create Lambda functions", "Step 3: Test failover"],
  "terraform_changes": "Update resource blocks to include Lambda functions, add auto-scaling policies, enable S3 intelligent-tiering",
  "rationale": "Best balance of cost savings with acceptable performance risk across services"
}}"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    response_text = response.content.strip()
    
    recommendation = {}
    try:
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx > start_idx:
            json_str = response_text[start_idx:end_idx]
            recommendation = json.loads(json_str)
        else:
            recommendation = json.loads(response_text)
    except json.JSONDecodeError:
        recommendation = {"analysis": response_text, "error": "Failed to parse JSON"}
    
    moderator_msg = f"MODERATOR DECISION:\n" + json.dumps(recommendation, indent=2)
    
    return {
        "final_recommendation": moderator_msg,
        "negotiation_history": state['negotiation_history'] + [moderator_msg]
    }

# --- Graph Logic ---

workflow = StateGraph(AgentState)

workflow.add_node("finops", finops_agent)
workflow.add_node("performance", performance_agent)
workflow.add_node("moderator", moderator_agent)

workflow.set_entry_point("finops")
workflow.add_edge("finops", "performance")
workflow.add_edge("performance", "moderator")
workflow.add_edge("moderator", END)

app = workflow.compile()

# --- Analysis Function ---

def analyze_infrastructure(terraform_file_path: str = None, cloud_stats: Dict[str, Any] = None):
    """Main function to analyze infrastructure and recommend optimizations"""
    
    # Load Terraform file or use default
    if terraform_file_path and os.path.exists(terraform_file_path):
        terraform_content = parse_terraform_file(terraform_file_path)
        print(f"Loaded Terraform from: {terraform_file_path}\n")
    else:
        # Default multi-service Terraform configuration
        terraform_content = """
resource "aws_instance" "web_server" {
  ami           = "ami-0c55b159cbfafe1f0"
  instance_type = "t3.medium"
  count         = 12
  tags = {
    Name = "WebServer"
  }
}

resource "aws_rds_instance" "postgres_db" {
  identifier     = "prod-database"
  engine         = "postgres"
  instance_class = "db.t3.large"
  allocated_storage = 100
  multi_az       = true
}

resource "aws_s3_bucket" "app_data" {
  bucket = "app-data-bucket"
}

resource "aws_dynamodb_table" "sessions" {
  name           = "sessions"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "session_id"
}

resource "aws_lambda_function" "api_handler" {
  filename      = "lambda.zip"
  function_name = "api_handler"
  role          = aws_iam_role.lambda_role.arn
  handler       = "index.handler"
  memory_size   = 512
}

resource "aws_elasticache_cluster" "redis_cache" {
  cluster_id           = "my-cache"
  engine               = "redis"
  node_type            = "cache.t3.micro"
  num_cache_nodes      = 1
}
"""
    
    # Use provided cloud stats or defaults
    if cloud_stats is None:
        cloud_stats = {
            "EC2": {
                "monthly_cost": "$2000",
                "average_cpu": "15%",
                "average_memory": "40%",
                "peak_qps": 500,
                "sla": "99.9%",
                "instance_type": "t3.medium",
                "count": 12
            },
            "RDS": {
                "monthly_cost": "$800",
                "cpu_utilization": "25%",
                "storage_used": "60GB of 100GB",
                "backup_storage": "$50/month",
                "instance_class": "db.t3.large",
                "multi_az": "enabled"
            },
            "S3": {
                "monthly_cost": "$200",
                "storage_used": "500GB",
                "requests_per_month": "2M PUT, 10M GET",
                "storage_class": "STANDARD",
                "versioning": "disabled"
            },
            "DynamoDB": {
                "monthly_cost": "$100",
                "billing_mode": "PAY_PER_REQUEST",
                "avg_request_units": "100 WCU, 500 RCU per second",
                "tables": 1
            },
            "Lambda": {
                "monthly_cost": "$50",
                "invocations_per_month": "10M",
                "average_duration": "200ms",
                "memory_configured": "512MB",
                "functions": 1
            },
            "ElastiCache": {
                "monthly_cost": "$30",
                "node_type": "cache.t3.micro",
                "engine": "redis",
                "cache_hit_ratio": "85%"
            }
        }
    
    print("=" * 70)
    print("MULTI-SERVICE AWS INFRASTRUCTURE OPTIMIZATION ANALYSIS")
    print("=" * 70)
    
    # Collect the final state
    final_state = {}
    stream_events = list(app.stream({
        "terraform_config": terraform_content,
        "cloud_stats": cloud_stats,
        "finops_proposals": [],
        "performance_feedback": [],
        "final_recommendation": None,
        "turn_count": 0,
        "negotiation_history": []
    }))
    
    # Extract all nested state values
    for event_dict in stream_events:
        for node_name, state_values in event_dict.items():
            if isinstance(state_values, dict):
                final_state.update(state_values)
    
    # Print FinOps proposals
    if final_state and final_state.get("finops_proposals"):
        proposals = final_state["finops_proposals"]
        print("\n" + "=" * 70)
        print("FINOPS PROPOSALS (Multi-Service Cost Optimization)")
        print("=" * 70)
        for i, proposal in enumerate(proposals, 1):
            print(f"\nProposal {i}:")
            if isinstance(proposal, dict):
                print(f"  Description: {proposal.get('description', 'N/A')}")
                print(f"  Estimated Savings: {proposal.get('estimated_savings', 'TBD')}")
                print(f"  Complexity: {proposal.get('complexity', 'Unknown')}")
                services = proposal.get('affected_services', [])
                if services:
                    print(f"  Affected Services: {', '.join(services)}")
            else:
                print(f"  {proposal}")
    
    # Print Performance feedback
    if final_state and final_state.get("performance_feedback"):
        feedback = final_state["performance_feedback"]
        print("\n" + "=" * 70)
        print("PERFORMANCE & SLA RISK ASSESSMENT")
        print("=" * 70)
        for i, fb in enumerate(feedback, 1):
            print(f"\nProposal {i} Risk Analysis:")
            if isinstance(fb, dict):
                print(f"  Risk Level: {fb.get('risk_level', 'Unknown')}")
                print(f"  Performance Impact: {fb.get('impact', 'N/A')}")
                print(f"  SLA Impact: {fb.get('sla_impact', 'N/A')}")
                print(f"  Mitigation: {fb.get('mitigation', 'N/A')}")
            else:
                print(f"  {fb}")
    
    # Print final recommendation
    if final_state and final_state.get("final_recommendation"):
        print("\n" + "=" * 70)
        print("FINAL RECOMMENDATION & ARCHITECTURE OPTIMIZATION")
        print("=" * 70)
        rec_data = final_state["final_recommendation"]
        
        try:
            start_idx = rec_data.find('{')
            end_idx = rec_data.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = rec_data[start_idx:end_idx]
                rec_json = json.loads(json_str)
                
                print("\nRANKED PROPOSALS (by Cost/Risk Trade-off):\n")
                for rank_entry in rec_json.get("ranking", []):
                    idx = rec_json['ranking'].index(rank_entry) + 1
                    print(f"  Rank {idx}: {rank_entry.get('description', 'N/A')}")
                    print(f"    - Savings: {rank_entry.get('savings', 'N/A')} | Risk: {rank_entry.get('risk', 'N/A')} | Score: {rank_entry.get('score', 'N/A')}/10")
                    services = rank_entry.get('affected_services', [])
                    if services:
                        print(f"    - Services: {', '.join(services)}")
                
                print(f"\nTOP RECOMMENDATION:\n  {rec_json.get('top_recommendation', 'N/A')}")
                
                if rec_json.get('implementation_steps'):
                    print(f"\nIMPLEMENTATION STEPS:")
                    for idx, step in enumerate(rec_json.get('implementation_steps', []), 1):
                        print(f"  {idx}. {step}")
                
                if rec_json.get('terraform_changes'):
                    print(f"\nTERRAFORM MODIFICATIONS:\n  {rec_json.get('terraform_changes', 'N/A')}")
                
                print(f"\nRATIONALE:\n  {rec_json.get('rationale', 'N/A')}")
        except Exception as e:
            print(f"Could not parse recommendation: {e}")
            print(rec_data)
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

# --- Main Execution ---

if __name__ == "__main__":
    print("Example 1: Analyzing default multi-service infrastructure...\n")
    analyze_infrastructure()
    