import os
import json
from typing import Dict, Any
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)


def architecture_agent(code_summarizer_output: Dict[str, Any], azure_metrics: Dict[str, Any], azure_cost: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze architecture using code summarizer output plus Azure runtime + cost signals.

    Returns a JSON-serializable dict shaped similarly to other agents in this repo.
    This agent must not invent resources/metrics/costs; it should be conservative when inputs are missing.
    """
    prompt = f"""You are a Senior Cloud Architecture Reviewer.

Your responsibility is STRICTLY architecture and reliability/scalability risk review.
You are NOT a cost optimizer (FinOps will handle that).
You are NOT a performance tuner (Performance agent will handle that).

You must base your analysis ONLY on the provided inputs.
Do NOT invent missing services, metrics, costs, resource names, or infrastructure components.
If something is unknown, set fields to null/[] and add a note in issues_detected.

You must return between 3 and 5 distinct architecture issues and between 3 and 5 concrete recommendations.

-----------------------------------
CODE SUMMARIZER OUTPUT (STRUCTURED JSON):
{json.dumps(code_summarizer_output or {}, ensure_ascii=False)}

-----------------------------------
AZURE RUNTIME METRICS (JSON):
{json.dumps(azure_metrics or {}, ensure_ascii=False)}

-----------------------------------
AZURE COST DATA (JSON):
{json.dumps(azure_cost or {}, ensure_ascii=False)}

-----------------------------------
Return ONLY a valid JSON object (no markdown, no commentary).

Schema:
{{
  "issues_detected": [string],
  "recommendations": [string],
  "architecture_style": string | null,
  "cloud_topology": {{
    "compute": string | null,
    "database": string | null,
    "storage": string | null,
    "networking": string | null,
    "caching_layer_present": boolean | null,
    "autoscaling_enabled": boolean | null
  }},
  "scalability_assessment": {{
    "horizontal_scaling_safe": boolean | null,
    "bottleneck_component": string | null,
    "scalability_risk_level": "Low" | "Medium" | "High" | "Unknown"
  }},
  "reliability_assessment": {{
    "single_point_of_failure_detected": boolean | null,
    "failover_strategy_detected": boolean | null,
    "resilience_score": number | null
  }}
}}
"""

    try:
        response = llm.invoke(prompt)
        response_text = (response.content or "").strip() if response else ""

        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        json_str = response_text[start_idx:end_idx] if start_idx != -1 and end_idx > start_idx else response_text
        parsed = json.loads(json_str)
        if isinstance(parsed, dict):
            # Enforce max 5 items for issues and recommendations (do not fabricate more)
            if isinstance(parsed.get("issues_detected"), list):
                parsed["issues_detected"] = parsed["issues_detected"][:5]
            if isinstance(parsed.get("recommendations"), list):
                parsed["recommendations"] = parsed["recommendations"][:5]

            parsed.setdefault("analysis_status", "completed")
            return parsed

        return {
            "analysis_status": "failed",
            "error": "LLM returned non-object JSON for architecture analysis",
            "raw_response": response_text[:2000],
        }
    except Exception as e:
        return {
            "analysis_status": "failed",
            "error": f"Architecture analysis failed: {str(e)[:200]}",
        }
