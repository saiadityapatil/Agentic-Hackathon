from data.github_extractor import extract_repo_code
import os
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.5)

def code_summarizer_agent(state, repo_url: str):
    code_files = extract_repo_code(repo_url)

    prompt = f"""
    You are a senior software architecture and infrastructure analyzer.

Your task is to analyze a backend codebase that may include both application source code and Terraform infrastructure files.

You are NOT allowed to provide explanations.
You MUST return only valid JSON.
Do not include markdown.
Do not include commentary.
Do not include analysis text.
Return a single JSON object.

The input will contain multiple source code files and optionally Terraform (.tf) files from a project.

You must extract structured architectural and infrastructure parameters.

Return the following JSON schema:

{{
"application": {{
"framework": string | null,
"language": string | null,
"project_structure": {{
"layered_architecture": boolean,
"separate_service_layer": boolean,
"repository_pattern": boolean,
"monolithic": boolean,
"microservice_ready": boolean,
"circular_imports_detected": boolean
}},
"api_design": {{
"route_count": integer,
"uses_dependency_injection": boolean,
"uses_pydantic_models": boolean,
"validation_present": boolean,
"pagination_supported": boolean,
"rate_limiting_present": boolean
}},
"concurrency": {{
"async_routes_count": integer,
"blocking_db_calls": boolean,
"background_tasks_used": boolean,
"threadpool_usage": boolean,
"global_state_used": boolean
}},
"database_architecture": {{
"database_used": boolean,
"orm_used": string | null,
"raw_sql_queries": integer,
"connection_pool_configured": boolean,
"transaction_management": boolean,
"n_plus_one_risk": boolean,
"long_running_queries_detected": boolean,
"index_usage_detected": boolean
}},
"caching": {{
"cache_layer_present": boolean,
"redis_used": boolean,
"in_memory_cache": boolean,
"cache_invalidation_strategy": string | null
}},
"security": {{
"auth_present": boolean,
"jwt_used": boolean,
"cors_configured": boolean,
"secrets_hardcoded": boolean,
"input_sanitization": boolean
}},
"devops": {{
"dockerized": boolean,
"ci_cd_pipeline": boolean,
"environment_variables_used": boolean,
"health_check_endpoint": boolean,
"logging_configured": boolean,
"structured_logging": boolean
}},
"observability": {{
"metrics_exposed": boolean,
"application_insights_integrated": boolean,
"logging_level_configurable": boolean,
"distributed_tracing": boolean
}},
"scalability": {{
"stateless_design": boolean,
"shared_session_state": boolean,
"file_storage_local": boolean,
"horizontal_scaling_safe": boolean,
"auto_scaling_ready": boolean
}},
"risk_indicators": {{
"heavy_select_queries": integer,
"large_payload_endpoints": integer,
"synchronous_external_calls": integer,
"cpu_intensive_loops": integer
}},
"architectural_concerns": [string]
}},
"infrastructure": {{
"cloud_provider": string | null,
"compute": {{
"app_service_present": boolean,
"container_based": boolean,
"serverless_used": boolean,
"vm_used": boolean,
"autoscaling_configured": boolean,
"instance_sku": string | null
}},
"database": {{
"managed_database": boolean,
"database_type": string | null,
"database_tier": string | null,
"private_networking_enabled": boolean,
"backup_configured": boolean
}},
"caching": {{
"redis_present": boolean,
"cache_sku": string | null
}},
"networking": {{
"vnet_configured": boolean,
"private_endpoints": boolean,
"public_access_enabled": boolean,
"load_balancer_present": boolean
}},
"security": {{
"managed_identity_used": boolean,
"key_vault_used": boolean,
"https_enforced": boolean,
"firewall_rules_defined": boolean
}},
"storage": {{
"storage_account_present": boolean,
"cdn_used": boolean
}},
"cost_risk_indicators": {{
"overprovisioned_compute": boolean,
"single_point_of_failure": boolean,
"no_autoscaling": boolean,
"public_database_exposed": boolean
}}
}}
}}

Extraction Rules:

Infer application framework from imports (e.g., FastAPI, Flask, Django).

Detect async usage via "async def".

Detect ORM via SQLAlchemy, Django ORM, etc.

Count raw SQL queries via SELECT/UPDATE/DELETE patterns.

Detect Docker via presence of Dockerfile.

Detect CI/CD via presence of GitHub Actions, Azure pipelines, or similar.

Detect caching via Redis, cache decorators, or in-memory cache structures.

Detect authentication via JWT libraries or login endpoints.

Mark "secrets_hardcoded" true if API keys/passwords appear in code.

Detect Terraform by presence of .tf files.

Infer cloud provider from Terraform provider blocks (azurerm, aws, google).

Detect compute/database/network/security resources from Terraform resource blocks.

If uncertain, default to false or null rather than guessing.

Return only JSON.
No extra text.

CODE FILES:
{code_files}
    """

    response = llm.invoke(prompt)
    response_text = response.content.strip()
    print(response_text)
    try:
        architectural_analysis = json.loads(response_text)
    except json.JSONDecodeError:
        architectural_analysis = {"error": "Failed to parse JSON response", "raw_response": response_text}
    return architectural_analysis

print(code_summarizer_agent({}, "https://github.com/Saiaditya004/agent-app.git"))