"""
Event emitter for agent completion tracking.
Provides a thread-safe way to emit agent events from the synchronous workflow.
"""

import asyncio
from typing import Optional, Dict, Callable
from queue import Queue
import threading


class AgentEventEmitter:
    """
    Emits events when agents complete.
    Thread-safe for use in synchronous workflow context.
    """
    
    def __init__(self):
        # Queue to store events from the synchronous workflow
        self.event_queue: Optional[Queue] = None
        # Callback functions for async handling
        self.callbacks: Dict[str, Callable] = {}
    
    def set_event_queue(self, queue: Queue):
        """Set the event queue for async processing."""
        self.event_queue = queue
    
    def emit_agent_started(self, agent_name: str):
        """Emit when an agent starts."""
        event = {
            "type": "agent_started",
            "agent": agent_name,
            "status": "in_progress"
        }
        if self.event_queue:
            self.event_queue.put(event)
        print(f"📤 Event emitted: {agent_name} started")
    
    def emit_agent_completed(self, agent_name: str, output: Dict):
        """Emit when an agent completes."""
        event = {
            "type": "agent_completed",
            "agent": agent_name,
            "status": "completed",
            "output": output
        }
        if self.event_queue:
            self.event_queue.put(event)
        print(f"📤 Event emitted: {agent_name} completed")
    
    def emit_agent_error(self, agent_name: str, error: str):
        """Emit when an agent encounters an error."""
        event = {
            "type": "agent_error",
            "agent": agent_name,
            "status": "error",
            "error": error
        }
        if self.event_queue:
            self.event_queue.put(event)
        print(f"📤 Event emitted: {agent_name} error - {error}")


# Global event emitter instance
agent_emitter = AgentEventEmitter()
