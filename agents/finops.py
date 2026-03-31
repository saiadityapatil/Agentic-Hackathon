import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)

def finops_agent(azure_metrics: dict, azure_cost: dict):
  prompt = f"""You are a Senior Azure FinOps Architect.

  Your responsibility is STRICTLY cost optimization.
  You are NOT a performance engineer.
  You are NOT an architecture reviewer.

  You must analyze Azure cost data in the context of runtime utilization metrics.
  Do not re-analyze code structure unless it directly impacts cost.

  You must make financially sound, production-safe recommendations.

  -----------------------------------
  AZURE RUNTIME METRICS:
  {azure_metrics}

  -----------------------------------
  AZURE COST DATA:
  {azure_cost}
  -----------------------------------

  Your Objective:

  Identify the TOP 3 cost optimization opportunities that:

  - Reduce unnecessary spending
  - Avoid degrading performance
  - Avoid increasing operational risk
  - Do NOT contradict obvious utilization signals

  -----------------------------------

  Mandatory Evaluation Rules:

  1. If CPU or utilization < 30% consistently → Likely overprovisioned.
  2. If utilization > 80% sustained → Scaling down is NOT allowed.
  3. If no autoscaling → Consider cost inefficiency during low traffic.
  4. If high database cost but moderate compute → Investigate DB tier mismatch.
  5. If storage is mostly inactive → Consider Cool/Archive tier.
  6. If long backup retention → Flag potential waste.
  7. If single-instance compute without scaling → Avoid aggressive downsizing.
  8. Never recommend changes that would likely increase performance bottlenecks.

  -----------------------------------

  For each recommendation provide:

  1. "issue_detected"
    Clear description of cost inefficiency.

  2. "recommendation"
    Concrete Azure action.
    (Example: downgrade App Service from P1v3 to B2, enable autoscale, move storage to Cool tier)

  3. "estimated_savings"
    - Provide percentage estimate.
    - Provide dollar estimate ONLY if cost data explicitly includes pricing.
    - If insufficient pricing info, provide percentage only.

  4. "risk_level"
    - Low (simple configuration change)
    - Medium (requires validation/testing)
    - High (architecture-level shift)

  5. "affected_services"
    List Azure services impacted.

  6. "confidence_level"
    - High (clear underutilization or waste)
    - Medium (moderate signal)
    - Low (limited cost clarity)

  -----------------------------------

  IMPORTANT CONSTRAINTS:

  - Do NOT hallucinate Azure pricing.
  - Do NOT invent metrics.
  - If data is insufficient, state conservative estimates.
  - Avoid generic advice like “optimize costs”.
  - Avoid more than 3 recommendations.

  -----------------------------------

  Return ONLY a valid JSON array.
  Start with [ and end with ].

  Each object must follow:

  {{
    "issue_detected": string,
    "recommendation": string,
    "estimated_savings": string,
    "risk_level": string,
    "affected_services": [string],
    "confidence_level": string
  }}

  No markdown.
  No commentary.
  No explanation outside JSON.
  """
  try:
      response = llm.invoke(prompt)
      response_text = response.content.strip() if response and response.content else ""
      
      if not response_text:
          return {
              "analysis_status": "failed",
              "error": "FinOps agent received empty response from LLM",
              "cost_inefficiencies": [],
          }
      
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
              return {
                  "analysis_status": "failed",
                  "error": "FinOps agent returned non-array JSON",
                  "raw_response": response_text[:2000],
                  "cost_inefficiencies": [],
              }
      except json.JSONDecodeError as e:
          return {
              "analysis_status": "failed",
              "error": f"Failed to parse FinOps JSON: {str(e)[:200]}",
              "raw_response": response_text[:2000],
              "cost_inefficiencies": [],
          }
      
      return {
          "cost_inefficiencies": proposals,
          "analysis_status": "completed"
      }
  except Exception as e:
      return {
          "analysis_status": "failed",
          "error": f"FinOps analysis failed: {str(e)[:200]}",
          "cost_inefficiencies": [],
      }