"""
Coordinator – task decomposition, auction-based allocation, consensus.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import time
import uuid
import threading

from .message_bus import MessageBus, Message
from .base_agent import BaseAgent


class Coordinator:
    """
    Central (but replaceable) orchestrator that:
    - registers agents
    - breaks high-level goals into sub-tasks
    - runs a simple first-price auction for allocation
    - collects results and writes them to memory
    """

    def __init__(self, bus: MessageBus, memory=None, ui=None):
        self.bus = bus
        self.memory = memory
        self.ui = ui
        self.agents: Dict[str, BaseAgent] = {}
        self.pending_tasks: Dict[str, dict] = {}
        self.results: Dict[str, Any] = {}
        self._lock = threading.RLock()

        self.bus.subscribe("coordinator", self._on_message)

    def register_agent(self, agent: BaseAgent) -> None:
        with self._lock:
            self.agents[agent.name] = agent
        agent.advertise()
        if self.ui:
            self.ui.update_progress(
                message=f"Agent registered: {agent.name} ({agent.role})",
                status="agents_ready",
            )

    def _on_message(self, msg: Message) -> None:
        payload = msg.payload or {}
        kind = payload.get("kind")

        if kind == "bid":
            self._handle_bid(payload)
        elif kind == "task_result":
            self._handle_result(payload)
        elif kind == "advertise":
            pass

    def _handle_bid(self, payload: dict) -> None:
        task_id = payload.get("task_id")
        with self._lock:
            task = self.pending_tasks.get(task_id)
            if not task or task.get("status") != "auction":
                return
            task.setdefault("bids", []).append(payload)

    def _handle_result(self, payload: dict) -> None:
        task_id = payload.get("task_id")
        with self._lock:
            self.results[task_id] = payload
            task = self.pending_tasks.get(task_id)
            if task:
                task["status"] = "completed" if payload.get("success") else "failed"
                task["result"] = payload

        if self.memory and payload.get("success"):
            self.memory.store_information(
                {
                    "event": "agent_task_complete",
                    "task_id": task_id,
                    "agent": payload.get("agent"),
                    "result": payload.get("result"),
                },
                task_signature=task_id,
                tags=["agent", "result"],
            )

        if self.ui:
            self.ui.update_progress(
                message=f"Task {task_id[:8]} completed by {payload.get('agent')}",
                status="task_complete",
            )

    def submit_goal(self, description: str, context: Optional[dict] = None) -> str:
        goal_id = str(uuid.uuid4())
        context = context or {}

        if self.ui:
            self.ui.update_progress(
                status="coordinating",
                message=f"Goal received: {description[:80]}",
            )

        pipeline = [
            {
                "name": "explore",
                "required_capabilities": ["explore"],
                "task": {
                    "description": description,
                    "n_candidates": 3,
                    **context,
                },
            },
            {
                "name": "optimize",
                "required_capabilities": ["optimize", "evolve"],
                "task": {
                    "description": description,
                    "generations": context.get("generations", 40),
                },
            },
            {
                "name": "critique",
                "required_capabilities": ["evaluate", "critique"],
                "task": {
                    "description": description,
                },
            },
            {
                "name": "store",
                "required_capabilities": ["store"],
                "task": {
                    "action": "store",
                    "task_signature": goal_id,
                    "tags": ["goal", "multi_agent"],
                },
            },
        ]

        results = []
        for step in pipeline:
            if step["name"] == "critique" and results:
                step["task"]["result"] = results[-1].get("result", {})
            if step["name"] == "store" and results:
                step["task"]["content"] = {
                    "goal": description,
                    "pipeline_results": results,
                }

            task_result = self._run_auction_and_execute(step)
            results.append(task_result)

        if self.memory:
            self.memory.store_information(
                {
                    "event": "goal_complete",
                    "goal_id": goal_id,
                    "description": description,
                    "results": results,
                },
                task_signature=goal_id,
                tags=["goal", "complete"],
            )

        if self.ui:
            self.ui.update_progress(
                status="goal_complete",
                message=f"Goal finished: {description[:60]}",
            )

        return goal_id

    def _run_auction_and_execute(self, step: dict, timeout: float = 8.0) -> dict:
        task_id = str(uuid.uuid4())
        with self._lock:
            self.pending_tasks[task_id] = {
                "status": "auction",
                "step": step["name"],
                "bids": [],
                "created": time.time(),
            }

        self.bus.publish(
            "broadcast",
            "Coordinator",
            {
                "kind": "task_offer",
                "task_id": task_id,
                "required_capabilities": step["required_capabilities"],
                "task": step["task"],
            },
        )

        time.sleep(0.6)
        with self._lock:
            bids = list(self.pending_tasks[task_id].get("bids", []))

        if not bids:
            return {
                "task_id": task_id,
                "step": step["name"],
                "success": False,
                "error": "no_bids",
            }

        winner = max(bids, key=lambda b: b.get("bid", 0))
        winner_name = winner["agent"]

        with self._lock:
            self.pending_tasks[task_id]["status"] = "assigned"
            self.pending_tasks[task_id]["winner"] = winner_name

        self.bus.publish(
            f"agent.{winner_name}",
            "Coordinator",
            {
                "kind": "task_assign",
                "task_id": task_id,
                "agent": winner_name,
                "task": step["task"],
            },
        )

        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if task_id in self.results:
                    return {
                        "task_id": task_id,
                        "step": step["name"],
                        "winner": winner_name,
                        **self.results[task_id],
                    }
            time.sleep(0.1)

        return {
            "task_id": task_id,
            "step": step["name"],
            "success": False,
            "error": "timeout",
            "winner": winner_name,
        }

    def status_report(self) -> dict:
        with self._lock:
            return {
                "agents": {
                    name: {
                        "role": a.role,
                        "status": a.status,
                        "load": a.load,
                        "capabilities": list(a.capabilities),
                    }
                    for name, a in self.agents.items()
                },
                "pending_tasks": len([t for t in self.pending_tasks.values() if t["status"] != "completed"]),
                "completed": len(self.results),
            }
