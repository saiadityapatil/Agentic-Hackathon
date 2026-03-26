"""
WebSocket connection manager for handling real-time agent updates.
Manages multiple concurrent WebSocket connections and broadcasts agent completion events.
"""

import json
import asyncio
from typing import Set, Dict
from fastapi import WebSocket


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        # Set of active WebSocket connections
        self.active_connections: Set[WebSocket] = set()
        # Lock for thread-safe operations
        self.lock = asyncio.Lock()
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.add(websocket)
        print(f"🔌 WebSocket connected. Total connections: {len(self.active_connections)}")
    
    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        self.active_connections.discard(websocket)
        print(f"🔌 WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict):
        """
        Broadcast a message to all connected clients.
        
        Args:
            message: Dictionary containing the message to send
        """
        if not self.active_connections:
            return
        
        # Add timestamp to message
        message_with_timestamp = {
            **message,
            "timestamp": asyncio.get_event_loop().time()
        }
        
        message_text = json.dumps(message_with_timestamp)
        
        # Send to all connected clients
        disconnected = set()
        for connection in self.active_connections:
            try:
                await connection.send_text(message_text)
            except Exception as e:
                print(f"⚠️ Failed to send WebSocket message: {str(e)}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        self.active_connections -= disconnected
    
    async def send_agent_completion(self, agent_name: str, output: Dict):
        """
        Send an agent completion message.
        
        Args:
            agent_name: Name of the agent that completed
            output: The output from the agent
        """
        message = {
            "type": "agent_completed",
            "agent": agent_name,
            "status": "completed",
            "output": output
        }
        await self.broadcast(message)
    
    async def send_agent_started(self, agent_name: str):
        """
        Send an agent started message.
        
        Args:
            agent_name: Name of the agent that started
        """
        message = {
            "type": "agent_started",
            "agent": agent_name,
            "status": "in_progress"
        }
        await self.broadcast(message)
    
    async def send_error(self, agent_name: str, error: str):
        """
        Send an error message.
        
        Args:
            agent_name: Name of the agent that encountered an error
            error: Error message
        """
        message = {
            "type": "agent_error",
            "agent": agent_name,
            "status": "error",
            "error": error
        }
        await self.broadcast(message)
    
    async def send_analysis_complete(self, result: Dict):
        """
        Send the final analysis completion message.
        
        Args:
            result: The complete analysis result
        """
        message = {
            "type": "analysis_completed",
            "status": "completed",
            "result": result
        }
        await self.broadcast(message)
    
    async def send_metrics(self, metrics: Dict):
        """
        Send Azure metrics data to frontend.
        
        Args:
            metrics: Dictionary containing Azure metrics and costs
        """
        message = {
            "type": "metrics_update",
            "status": "available",
            "data": metrics
        }
        await self.broadcast(message)
    
    async def send_message(self, message_type: str, data: Dict):
        """
        Send a custom message to all clients.
        
        Args:
            message_type: Type of message
            data: Message data
        """
        message = {
            "type": message_type,
            **data
        }
        await self.broadcast(message)


# Global connection manager instance
manager = ConnectionManager()
