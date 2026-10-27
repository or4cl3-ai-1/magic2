"""
Base Agent – lightweight BDI-inspired agent.
Beliefs, Desires, Intentions + capability advertisement.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set
import time
import uuid

from .message_bus import MessageBus, Message


class BaseAgent(ABC):
    """
    Minimal autonomous agent that can:
    - advertise capabilities
    - receive tasks via the message bus
    - maintain simple beliefs / current intention
    - report status and results
    """

    def __init__(
        self,
        name: str,
        capabilities: List[str],
        bus: MessageBus,
        role: str = "generic",
    ):
        self.id = str(uuid.uuid4())[:8]
        self.name = name
        self.role = role
        self.capabilities: Set[str] = set(capabilities)
        self.bus = bus
        self.beliefs: Dict[str, Any] = {}
        self.current_intention: Optional[str] = None
        self.status = "idle"  # idle | busy | waiting | error
        self.load = 0.0       # 0.0 – 1.0
        self.last_result: Any = None
        self.created_at = time.time()

        self.bus.subscribe(f"agent.{self.name}", self._on_message)
        self.bus.subscribe("broadcast", self._on_message)
        self.bus.subscribe(f"role.{self.role}", self._on_message)

    def _on_message(self, msg: Message) -> None:
        if msg.sender == self.name:
            return
        payload = msg.payload or {}
        kind = payload.get("kind")

        if kind == "task_offer":
            self._handle_task_offer(payload)
        elif kind == "task_assign":
            self._handle_task_assign(payload)
        elif kind == "query":
            self._handle_query(payload)
        elif kind == "update_belief":
            self.beliefs.update(payload.get("beliefs", {}))
        else:
            self.on_message(msg)

    def _handle_task_offer(self, payload: dict) -> None:
        task_id = payload.get("task_id")
        required = set(payload.get("required_capabilities", []))
        if required and not required.issubset(self.capabilities):
            return
        if self.status != "idle" or self.load > 0.8:
            return
        bid = self.compute_bid(payload)
        self.bus.publish(
            "coordinator",
            self.name,
            {
                "kind": "bid",
                "task_id": task_id,
                "agent": self.name,
                "bid": bid,
                "capabilities": list(self.capabilities),
            },
        )

    def _handle_task_assign(self, payload: dict) -> None:
        if payload.get("agent") != self.name:
            return
        self.status = "busy"
        self.current_intention = payload.get("task_id")
        self.load = min(1.0, self.load + 0.4)
        try:
            result = self.execute(payload.get("task", {}))
            self.last_result = result
            self.bus.publish(
                "coordinator",
                self.name,
                {
                    "kind": "task_result",
                    "task_id": self.current_intention,
                    "agent": self.name,
                    "success": True,
                    "result": result,
                },
            )
        except Exception as e:
            self.bus.publish(
                "coordinator",
                self.name,
                {
                    "kind": "task_result",
                    "task_id": self.current_intention,
                    "agent": self.name,
                    "success": False,
                    "error": str(e),
                },
            )
        finally:
            self.status = "idle"
            self.current_intention = None
            self.load = max(0.0, self.load - 0.4)

    def _handle_query(self, payload: dict) -> None:
        self.bus.publish(
            payload.get("reply_to", "coordinator"),
            self.name,
            {
                "kind": "query_response",
                "agent": self.name,
                "status": self.status,
                "load": self.load,
                "capabilities": list(self.capabilities),
                "beliefs": self.beliefs,
            },
        )

    def compute_bid(self, task_payload: dict) -> float:
        base = 1.0 - self.load
        required = set(task_payload.get("required_capabilities", []))
        overlap = len(required & self.capabilities)
        return base + 0.3 * overlap

    def advertise(self) -> None:
        self.bus.publish(
            "coordinator",
            self.name,
            {
                "kind": "advertise",
                "agent": self.name,
                "role": self.role,
                "capabilities": list(self.capabilities),
                "status": self.status,
                "load": self.load,
            },
        )

    @abstractmethod
    def execute(self, task: dict) -> Any:
        ...

    def on_message(self, msg: Message) -> None:
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, status={self.status}, load={self.load:.2f})"
