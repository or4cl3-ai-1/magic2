"""
M.A.G.I.C. Multi-Agent Coordination Layer
"""

from .base_agent import BaseAgent
from .specialist_agents import ExplorerAgent, OptimizerAgent, CriticAgent, MemoryKeeperAgent
from .coordinator import Coordinator
from .message_bus import MessageBus

__all__ = [
    "BaseAgent",
    "ExplorerAgent",
    "OptimizerAgent",
    "CriticAgent",
    "MemoryKeeperAgent",
    "Coordinator",
    "MessageBus",
]
