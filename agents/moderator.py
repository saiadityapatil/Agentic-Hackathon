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


