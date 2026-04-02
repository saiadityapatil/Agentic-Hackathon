import json
import asyncio
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from graph import workflow
from state import State
from websocket_manager import manager
from event_emitter import agent_emitter
from metrics_extractor import get_metrics_extractor
from data.azure_extractor import create_azure_extractor

# Initialize FastAPI app
app = FastAPI(
    title="Code Analysis Engine",
    description="Analyzes code repositories using multi-agent architecture",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (development) - restrict in production
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


class MetricsRequest(BaseModel):
    """Request model for /metrics endpoint"""
    azure_client_id: Optional[str] = None
    azure_client_secret: Optional[str] = None
    azure_tenant_id: Optional[str] = None
    azure_subscription_id: Optional[str] = None
    resource_group_name: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "azure_client_id": "your-client-id",
                "azure_client_secret": "your-client-secret",
                "azure_tenant_id": "your-tenant-id",
                "azure_subscription_id": "your-subscription-id",
                "resource_group_name": "your-resource-group"
            }
        }


class AnalysisRequest(BaseModel):
    """Request model for /analyze endpoint"""
    repo_url: str
    azure_client_id: Optional[str] = None
    azure_client_secret: Optional[str] = None
    azure_tenant_id: Optional[str] = None
    azure_subscription_id: Optional[str] = None
    resource_group_name: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "repo_url": "https://github.com/your-username/your-repo",
                "azure_client_id": "your-client-id",
                "azure_client_secret": "your-client-secret",
                "azure_tenant_id": "your-tenant-id",
                "azure_subscription_id": "your-subscription-id",
                "resource_group_name": "your-resource-group"
            }
        }


class AnalysisResponse(BaseModel):
    """Response model for /analyze endpoint"""
    status: str
    repo_url: str
    code_summarizer: dict
    architecture: dict
    performance: dict
    finops: dict
    moderator: dict
    metrics: dict
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "repo_url": "https://github.com/...",
                "code_summarizer": {"framework": "...", "language": "..."},
                "architecture": {"issues_detected": [...], "recommendations": [...]},
                "performance": {"issues_detected": [...], "recommendations": [...]},
                "finops": {"cost_inefficiencies": [...], "recommendations": [...]},
                "moderator": {"conflicts_detected": [...], "ranked_recommendations": [...]},
                "metrics": {"app_service": {...}, "sql_database": {...}, "storage": {...}, "costs": {...}}
            }
        }


# Thread pool for running blocking workflow operations
executor = ThreadPoolExecutor(max_workers=4)


async def process_workflow_events(event_queue: Queue):
    """
    Process events from the workflow and broadcast them to WebSocket clients.
    Runs concurrently while the workflow executes.
    
    Args:
        event_queue: Queue containing events from the workflow
    """
    while True:
        try:
            # Non-blocking check for events
            if not event_queue.empty():
                event = event_queue.get_nowait()
                
                if event["type"] == "agent_started":
                    await manager.send_agent_started(event["agent"])
                elif event["type"] == "agent_completed":
                    await manager.send_agent_completion(event["agent"], event["output"])
                elif event["type"] == "agent_error":
                    await manager.send_error(event["agent"], event["error"])
                elif event["type"] == "analysis_completed":
                    await manager.send_analysis_complete(event["result"])
            else:
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error processing workflow events: {str(e)}")
            await asyncio.sleep(0.1)


async def run_workflow_with_events(repo_url: str, azure_credentials: Optional[dict], resource_group_name: Optional[str], event_queue: Queue):
    """
    Run the workflow in a thread pool and emit events to the queue.
    
    Args:
        repo_url: Repository URL to analyze
        azure_credentials: Optional Azure service principal credentials
        resource_group_name: Optional Azure resource group name
        event_queue: Queue to emit events to
        
    Returns:
        The workflow result
    """
    loop = asyncio.get_event_loop()
    
    def workflow_execution():
        # Set up the event emitter with the queue
        agent_emitter.set_event_queue(event_queue)
        
        # Initialize state
        initial_state: State = {
            "repo_url": repo_url,
            "azure_credentials": azure_credentials,
            "resource_group_name": resource_group_name,
            "azure_metrics": None,
            "azure_costs": None,
            "code_summarizer_output": None,
            "architecture_output": None,
            "performance_output": None,
            "finops_output": None,
            "moderator_output": None,
            "final_analysis": None
        }
        
        print(f"\n{'='*60}")
        print(f"🚀 Starting Analysis for: {repo_url}")
        if azure_credentials and resource_group_name:
            print(f"🔵 Azure Resource Group: {resource_group_name}")
        print(f"{'='*60}\n")
        
        # Execute workflow
        result = workflow.invoke(initial_state)
        
        print(f"\n{'='*60}")
        print(f"✅ Analysis Complete for: {repo_url}")
        print(f"{'='*60}\n")
        
        # Queue final analysis complete event
        event_queue.put({
            "type": "analysis_completed",
            "result": {
                "status": "success",
                "repo_url": repo_url,
                "code_summarizer": result.get('code_summarizer_output') or {},
                "architecture": result.get('architecture_output') or {},
                "performance": result.get('performance_output') or {},
                "finops": result.get('finops_output') or {},
                "moderator": result.get('moderator_output') or {}
            }
        })
        
        return result
    
    # Run workflow in thread pool
    result = await loop.run_in_executor(executor, workflow_execution)
    return result


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Code Analysis Engine",
        "version": "1.0.0",
        "websocket": "ws://localhost:8000/ws"
    }


@app.post("/metrics", tags=["Metrics"])
async def get_metrics(request: Optional[MetricsRequest] = None):
    """
    Get current Azure metrics and cost data.
    
    If Azure credentials are provided, fetches live data from Azure.
    Otherwise, returns data from static JSON files.
    
    Args:
        request: Optional MetricsRequest with Azure credentials
    
    Returns:
        Dictionary containing:
        - app_service: CPU/memory utilization and scaling info
        - sql_database: DTU utilization and monthly cost
        - storage: Storage usage and costs
        - costs: Total monthly cost breakdown
    """
    try:
        # Check if Azure credentials are provided
        if request and all([
            request.azure_client_id, 
            request.azure_client_secret,
            request.azure_tenant_id, 
            request.azure_subscription_id,
            request.resource_group_name
        ]):
            # Fetch live Azure data
            print(f"🔵 Fetching live Azure metrics for resource group: {request.resource_group_name}")
            
            try:
                extractor = create_azure_extractor(
                    client_id=request.azure_client_id,
                    client_secret=request.azure_client_secret,
                    tenant_id=request.azure_tenant_id,
                    subscription_id=request.azure_subscription_id,
                    resource_group_name=request.resource_group_name
                )
                
                azure_metrics = extractor.extract_all_metrics()
                azure_costs = extractor.extract_all_costs()
                
                # Merge metrics and costs
                metrics_data = {
                    "app_service": {**azure_metrics.get("app_service", {}), **azure_costs.get("app_service", {})},
                    "sql_database": {**azure_metrics.get("sql_database", {}), **azure_costs.get("sql_database", {})},
                    "storage": {**azure_metrics.get("storage_account", {}), **azure_costs.get("storage_account", {})},
                    "costs": {
                        "total_monthly_cost": "$0",
                        "services_breakdown": azure_costs
                    }
                }
                
                # Calculate total cost
                total_cost = 0.0
                for service_costs in azure_costs.values():
                    cost_str = service_costs.get("cost_per_month", "$0")
                    try:
                        cost_value = float(cost_str.replace("$", "").replace(",", ""))
                        total_cost += cost_value
                    except (ValueError, AttributeError):
                        pass
                
                metrics_data["costs"]["total_monthly_cost"] = f"${total_cost:.2f}"
                
                print("✅ Successfully fetched live Azure metrics")
                
                return {
                    "status": "success",
                    "source": "azure_live",
                    "data": metrics_data
                }
            except Exception as e:
                print(f"⚠️ Error fetching live Azure data: {str(e)}")
                print("📁 Falling back to static JSON files")
        
        # Fall back to static JSON files
        metrics_extractor = get_metrics_extractor()
        metrics_data = metrics_extractor.get_all_metrics()
        
        return {
            "status": "success",
            "source": "static_files",
            "data": metrics_data
        }
    except Exception as e:
        print(f"❌ Error fetching metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching metrics: {str(e)}"
        )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time agent completion updates and metrics.
    Clients connect here to receive:
    - Live updates as agents complete
    - Azure metrics and cost data
    """
    await manager.connect(websocket)
    try:
        # Send initial metrics data to the newly connected client
        try:
            metrics_extractor = get_metrics_extractor()
            metrics_data = metrics_extractor.get_all_metrics()
            await manager.send_metrics(metrics_data)
            print("📊 Metrics sent to newly connected client")
        except Exception as e:
            print(f"⚠️ Error sending metrics: {str(e)}")
        
        # Keep connection open and handle any incoming messages
        while True:
            data = await websocket.receive_text()
            # Echo back or handle ping/pong
            if data.lower() == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {str(e)}")
        await manager.disconnect(websocket)


@app.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_repository(request: AnalysisRequest):
    """
    Analyze a GitHub repository using the multi-agent workflow.
    
    Flow:
    1. Code Summarizer: Extracts and summarizes code structure
    2. Architecture Agent: Analyzes architecture (uses code summarizer output)
    3. Performance Agent: Identifies performance bottlenecks (parallel)
    4. FinOps Agent: Analyzes cost optimization (parallel)
    5. Moderator Agent: Synthesizes all outputs into actionable recommendations
    
    Optionally provide Azure credentials to fetch live Azure metrics and costs.
    
    Note: Real-time agent completion updates are sent via WebSocket at /ws
    
    Args:
        request: AnalysisRequest with repo_url and optional Azure credentials
        
    Returns:
        AnalysisResponse with complete analysis results
    """
    try:
        # Validate and use provided repo_url
        repo_url = request.repo_url.strip() if request.repo_url else ""
        
        if not repo_url:
            raise HTTPException(
                status_code=400,
                detail="repo_url is required. Must be a valid GitHub repository URL."
            )
        
        # Extract Azure credentials if provided
        azure_credentials = None
        resource_group_name = None
        
        if all([request.azure_client_id, request.azure_client_secret, 
                request.azure_tenant_id, request.azure_subscription_id, 
                request.resource_group_name]):
            azure_credentials = {
                "client_id": request.azure_client_id,
                "client_secret": request.azure_client_secret,
                "tenant_id": request.azure_tenant_id,
                "subscription_id": request.azure_subscription_id
            }
            resource_group_name = request.resource_group_name
            print(f"🔵 Azure credentials provided for resource group: {resource_group_name}")
        else:
            print("📁 No Azure credentials provided, using static data")
        
        # Create an event queue for this analysis
        event_queue: Queue = Queue()
        
        # Run workflow and process events concurrently
        workflow_task = asyncio.create_task(
            run_workflow_with_events(repo_url, azure_credentials, resource_group_name, event_queue)
        )
        event_task = asyncio.create_task(process_workflow_events(event_queue))
        
        # Wait for workflow to complete
        result = await workflow_task
        
        # Cancel event processing once workflow is done
        event_task.cancel()
        
        # Extract individual agent responses
        code_summarizer = result.get('code_summarizer_output') or {}
        architecture = result.get('architecture_output') or {}
        performance = result.get('performance_output') or {}
        finops = result.get('finops_output') or {}
        moderator = result.get('moderator_output') or {}
        
        # Get metrics data from workflow result (already fetched during workflow execution)
        azure_metrics = result.get('azure_metrics') or {}
        azure_costs = result.get('azure_costs') or {}
        
        # If we have live Azure data, format it properly
        if azure_metrics and azure_costs:
            metrics_data = {
                "app_service": {**azure_metrics.get("app_service", {}), **azure_costs.get("app_service", {})},
                "sql_database": {**azure_metrics.get("sql_database", {}), **azure_costs.get("sql_database", {})},
                "storage": {**azure_metrics.get("storage_account", {}), **azure_costs.get("storage_account", {})},
                "costs": {
                    "total_monthly_cost": "$0",
                    "services_breakdown": azure_costs
                }
            }
            
            # Calculate total cost
            total_cost = 0.0
            for service_costs in azure_costs.values():
                cost_str = service_costs.get("cost_per_month", "$0")
                try:
                    cost_value = float(cost_str.replace("$", "").replace(",", ""))
                    total_cost += cost_value
                except (ValueError, AttributeError):
                    pass
            
            metrics_data["costs"]["total_monthly_cost"] = f"${total_cost:.2f}"
        else:
            # Fallback to static metrics if no Azure data in result
            try:
                metrics_extractor = get_metrics_extractor()
                metrics_data = metrics_extractor.get_all_metrics()
            except Exception as e:
                print(f"⚠️ Error fetching metrics: {str(e)}")
                metrics_data = {}
        
        # Return structured response with all agent outputs and metrics
        return AnalysisResponse(
            status="success",
            repo_url=repo_url,
            code_summarizer=code_summarizer,
            architecture=architecture,
            performance=performance,
            finops=finops,
            moderator=moderator,
            metrics=metrics_data
        )
        
    except HTTPException as http_ex:
        raise http_ex
    except json.JSONDecodeError as json_ex:
        print(f"❌ JSON parsing error: {str(json_ex)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing analysis results: {str(json_ex)}"
        )
    except Exception as e:
        print(f"❌ Workflow execution error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Workflow execution failed: {str(e)}"
        )


@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Code Analysis Engine",
        "version": "1.0.0",
        "endpoints": {
            "health": "GET /health",
            "analyze": "POST /analyze",
            "websocket": "WS /ws",
            "docs": "GET /docs",
            "openapi": "GET /openapi.json"
        },
        "websocket_usage": {
            "description": "Connect to /ws to receive real-time agent completion updates",
            "message_types": [
                "agent_started - emitted when an agent starts running",
                "agent_completed - emitted when an agent finishes with its output",
                "agent_error - emitted when an agent encounters an error",
                "analysis_completed - emitted when the full analysis is done"
            ],
            "example": {
                "agent_completed": {
                    "type": "agent_completed",
                    "agent": "code_summarizer",
                    "status": "completed",
                    "output": { }
                }
            }
        },
        "workflow": {
            "description": "Multi-agent code analysis workflow",
            "flow": [
                "Code Summarizer (Sequential)",
                "Architecture Agent (Sequential - uses code summarizer output)",
                "Performance Agent (Parallel)",
                "FinOps Agent (Parallel)",
                "Moderator Agent (Final synthesis)"
            ]
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
