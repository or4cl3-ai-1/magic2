"""
Simple in-process publish/subscribe message bus.
Later can be swapped for Redis, ZeroMQ, or a proper event backbone.
"""

from __future__ import annotations
from collections import defaultdict
from typing import Any, Callable, Dict, List
import threading
import time
import uuid


class Message:
    def __init__(self, topic: str, sender: str, payload: Any, msg_id: str | None = None):
        self.id = msg_id or str(uuid.uuid4())
        self.topic = topic
        self.sender = sender
        self.payload = payload
        self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "sender": self.sender,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


class MessageBus:
    """Thread-safe in-process pub/sub."""

    def __init__(self):
        self._subscribers: Dict[str, List[Callable[[Message], None]]] = defaultdict(list)
        self._lock = threading.RLock()
        self._history: List[Message] = []
        self._max_history = 500

    def subscribe(self, topic: str, callback: Callable[[Message], None]) -> None:
        with self._lock:
            self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable[[Message], None]) -> None:
        with self._lock:
            if callback in self._subscribers[topic]:
                self._subscribers[topic].remove(callback)

    def publish(self, topic: str, sender: str, payload: Any) -> Message:
        msg = Message(topic, sender, payload)
        with self._lock:
            self._history.append(msg)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]
            callbacks = list(self._subscribers.get(topic, []))
            callbacks += list(self._subscribers.get("*", []))

        for cb in callbacks:
            try:
                cb(msg)
            except Exception as e:
                print(f"[MessageBus] subscriber error on '{topic}': {e}")
        return msg

    def get_history(self, topic: str | None = None, limit: int = 50) -> List[dict]:
        with self._lock:
            msgs = self._history
            if topic:
                msgs = [m for m in msgs if m.topic == topic]
            return [m.to_dict() for m in msgs[-limit:]]
