import os
import json
from typing import Dict, Any
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)


def performance_agent(code_summarizer_output: Dict[str, Any], azure_metrics: Dict[str, Any], azure_cost: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze performance using code summarizer output plus Azure runtime + cost signals.

    Returns a JSON-serializable dict shaped similarly to the rest of this repo.
    This agent must not invent metrics/latencies/throughput; it should be conservative when inputs are missing.
    """
    prompt = f"""You are a Performance SRE.

Your responsibility is STRICTLY performance risk identification and safe mitigations.
You are NOT a cost optimizer (FinOps will handle that).
You are NOT an architecture reviewer (Architecture agent will handle that).

You must base your analysis ONLY on the provided inputs.
Do NOT invent metrics, response times, or throughput numbers.
If metrics do not contain latency/throughput, leave those fields null.

You must return between 3 and 5 distinct performance issues and between 3 and 5 concrete recommendations.

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
  "utilization_summary": {{
    "compute_cpu": string | null,
    "compute_memory": string | null,
    "compute_instances": number | null,
    "autoscaling_enabled": boolean | null,
    "database_utilization": string | null
  }},
  "bottlenecks": [string],
  "sla_risks": [string],
  "response_time_analysis": {{
    "average_response_time_ms": number | null,
    "p95_response_time_ms": number | null,
    "p99_response_time_ms": number | null
  }},
  "throughput": {{
    "requests_per_second": number | null,
    "current_load_percentage": number | null
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
            "error": "LLM returned non-object JSON for performance analysis",
            "raw_response": response_text[:2000],
        }
    except Exception as e:
        return {
            "analysis_status": "failed",
            "error": f"Performance analysis failed: {str(e)[:200]}",
        }
