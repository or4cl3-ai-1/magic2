"""
Specialist Agents – concrete roles that collaborate via the Coordinator.
"""

from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np

from .base_agent import BaseAgent
from .message_bus import MessageBus


class ExplorerAgent(BaseAgent):
    """Scouts the problem space / generates candidate approaches."""

    def __init__(self, bus: MessageBus, name: str = "Explorer"):
        super().__init__(
            name=name,
            capabilities=["explore", "generate_candidates", "survey"],
            bus=bus,
            role="explorer",
        )

    def execute(self, task: dict) -> Any:
        description = task.get("description", "")
        n_candidates = task.get("n_candidates", 3)
        paradigms = ["deep_learning", "reinforcement_learning", "meta_learning", "ensemble_learning"]
        sizes = [16, 32, 64, 128]
        candidates = []
        rng = np.random.default_rng()
        for i in range(n_candidates):
            candidates.append({
                "id": f"cand-{i}",
                "paradigm": str(rng.choice(paradigms)),
                "hidden_size": int(rng.choice(sizes)),
                "rationale": f"Exploration variant {i} for task: {description[:60]}",
            })
        self.beliefs["last_candidates"] = candidates
        return {"candidates": candidates, "agent": self.name}


class OptimizerAgent(BaseAgent):
    """Runs the genetic algorithm / trains networks."""

    def __init__(self, bus: MessageBus, name: str = "Optimizer", ga=None, nn=None):
        super().__init__(
            name=name,
            capabilities=["optimize", "evolve", "train"],
            bus=bus,
            role="optimizer",
        )
        self.ga = ga
        self.nn = nn

    def execute(self, task: dict) -> Any:
        if self.ga is None or self.nn is None:
            return {"error": "GA or NN not injected", "agent": self.name}

        generations = task.get("generations", 30)
        result = self.ga.evolve(generations=generations)
        self.beliefs["last_best_fitness"] = result["best_fitness"]
        return {
            "best_fitness": result["best_fitness"],
            "generations_run": result["generations_run"],
            "agent": self.name,
        }


class CriticAgent(BaseAgent):
    """Evaluates results and provides feedback / quality scores."""

    def __init__(self, bus: MessageBus, name: str = "Critic"):
        super().__init__(
            name=name,
            capabilities=["evaluate", "critique", "score"],
            bus=bus,
            role="critic",
        )

    def execute(self, task: dict) -> Any:
        result = task.get("result", {})
        fitness = result.get("best_fitness", 0.0)
        if fitness >= 0.9:
            verdict = "excellent"
            score = 0.95
        elif fitness >= 0.7:
            verdict = "good"
            score = 0.8
        elif fitness >= 0.5:
            verdict = "acceptable"
            score = 0.6
        else:
            verdict = "needs_improvement"
            score = 0.3
        critique = {
            "verdict": verdict,
            "score": score,
            "fitness": fitness,
            "suggestion": "Increase population or generations" if score < 0.7 else "Looks solid",
            "agent": self.name,
        }
        self.beliefs["last_critique"] = critique
        return critique


class MemoryKeeperAgent(BaseAgent):
    """Interfaces with the MemoryModule – stores and retrieves experiences."""

    def __init__(self, bus: MessageBus, memory=None, name: str = "MemoryKeeper"):
        super().__init__(
            name=name,
            capabilities=["store", "retrieve", "search_memory"],
            bus=bus,
            role="memory_keeper",
        )
        self.memory = memory

    def execute(self, task: dict) -> Any:
        action = task.get("action", "store")
        if self.memory is None:
            return {"error": "MemoryModule not injected", "agent": self.name}

        if action == "store":
            self.memory.store_information(
                task.get("content"),
                task_signature=task.get("task_signature", "agent"),
                tags=task.get("tags", ["agent"]),
            )
            return {"stored": True, "agent": self.name}
        elif action == "retrieve":
            entries = self.memory.retrieve_information(
                task_signature=task.get("task_signature"),
                limit=task.get("limit", 10),
            )
            return {"entries": entries, "agent": self.name}
        else:
            return {"error": f"Unknown action {action}", "agent": self.name}
