"""
M.A.G.I.C. Core – Dynamic Layer Generator
Produces layer configurations that can be evolved by the genetic algorithm.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import random


class DynamicLayer:
    """
    Generates neural-network layer specifications on demand.
    Supports multiple layer types so the GA can evolve architecture diversity.
    """

    SUPPORTED_TYPES = ["dense", "dropout", "batch_norm"]  # extensible later

    def __init__(
        self,
        layer_size: int = 32,
        activation: str = "relu",
        layer_type: str = "dense",
    ):
        self.layer_size = layer_size
        self.activation = activation
        self.layer_type = layer_type if layer_type in self.SUPPORTED_TYPES else "dense"

    def generate(self) -> Dict[str, Any]:
        """Return a serialisable layer specification."""
        spec = {
            "type": self.layer_type,
            "size": self.layer_size,
            "activation": self.activation,
        }
        if self.layer_type == "dropout":
            spec["rate"] = random.uniform(0.1, 0.5)
        return spec

    def mutate(self, strength: float = 0.3) -> "DynamicLayer":
        """Return a mutated copy – useful when the GA operates on layer genomes."""
        new_size = max(4, int(self.layer_size * (1 + random.uniform(-strength, strength))))
        acts = ["relu", "tanh", "sigmoid", "linear"]
        new_act = random.choice(acts) if random.random() < strength else self.activation
        new_type = random.choice(self.SUPPORTED_TYPES) if random.random() < strength * 0.5 else self.layer_type
        return DynamicLayer(layer_size=new_size, activation=new_act, layer_type=new_type)

    def __repr__(self) -> str:
        return f"DynamicLayer(type={self.layer_type}, size={self.layer_size}, act={self.activation})"
