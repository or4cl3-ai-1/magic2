"""
Architecture Genome – encode / decode neural-network topologies
so the GeneticAlgorithm can evolve structure, not just weights.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import random

from neural_network import NeuralNetwork


# Discrete choices the genome can select
LAYER_TYPES = ["dense", "dropout"]          # extensible later
ACTIVATIONS = ["relu", "tanh", "sigmoid", "linear"]
SIZE_CHOICES = [8, 16, 32, 64, 128, 256]

# Genome layout (fixed max depth for simplicity & speed)
MAX_LAYERS = 5
# Per layer: type_idx (1) + size_idx (1) + act_idx (1) = 3 genes
# Plus 1 gene for actual depth → total genes = 1 + MAX_LAYERS * 3
GENOME_LENGTH = 1 + MAX_LAYERS * 3


def random_architecture_genome() -> np.ndarray:
    """Create a random valid architecture genome (real-valued in [0, 1])."""
    return np.random.uniform(0.0, 1.0, size=GENOME_LENGTH)


def decode_genome(genome: np.ndarray) -> List[Dict[str, Any]]:
    """
    Turn a flat genome into a list of layer specifications.
    genome[0] controls depth; subsequent triples control each layer.
    """
    g = np.clip(genome, 0.0, 1.0)
    depth = 1 + int(g[0] * MAX_LAYERS)          # 1 … MAX_LAYERS
    depth = min(depth, MAX_LAYERS)

    layers = []
    for i in range(depth):
        base = 1 + i * 3
        type_idx = int(g[base] * len(LAYER_TYPES)) % len(LAYER_TYPES)
        size_idx = int(g[base + 1] * len(SIZE_CHOICES)) % len(SIZE_CHOICES)
        act_idx = int(g[base + 2] * len(ACTIVATIONS)) % len(ACTIVATIONS)

        spec = {
            "type": LAYER_TYPES[type_idx],
            "size": SIZE_CHOICES[size_idx],
            "activation": ACTIVATIONS[act_idx],
        }
        if spec["type"] == "dropout":
            # map size gene to a dropout rate instead
            spec["rate"] = 0.1 + 0.4 * g[base + 1]
            spec["size"] = None
        layers.append(spec)
    return layers


def build_network_from_genome(
    genome: np.ndarray,
    input_shape: Tuple[int, ...] = (8,),
    output_shape: int = 1,
    backend: str = "numpy",
) -> NeuralNetwork:
    """Instantiate a NeuralNetwork whose topology matches the genome."""
    layers = decode_genome(genome)
    nn = NeuralNetwork(input_shape=input_shape, output_shape=output_shape, backend=backend)
    for spec in layers:
        if spec["type"] == "dense":
            nn.add_layer(spec["size"], spec["activation"])
        elif spec["type"] == "dropout":
            # NumPy backend ignores dropout at inference; still record it
            nn.add_layer(max(8, int(32 * (1 - spec.get("rate", 0.2)))), "relu")
    # Ensure at least a minimal network
    if not nn.layers:
        nn.add_layer(32, "relu")
    return nn


def complexity_penalty(layers: List[Dict[str, Any]], alpha: float = 0.002) -> float:
    """Mild penalty so the search does not just grow huge nets."""
    total_params = 0
    prev = 8  # assume default input
    for spec in layers:
        if spec["type"] == "dense" and spec.get("size"):
            total_params += prev * spec["size"] + spec["size"]
            prev = spec["size"]
    return alpha * total_params


def architecture_fitness_factory(
    x: np.ndarray,
    y: np.ndarray,
    input_shape: Tuple[int, ...] = (8,),
    epochs: int = 5,
    complexity_alpha: float = 0.0015,
):
    """
    Returns a fitness function that:
    1. Builds a network from the genome
    2. Does a short training run
    3. Returns 1 / (1 + mse) – complexity_penalty
    """

    def fitness(genome: np.ndarray) -> float:
        try:
            nn = build_network_from_genome(genome, input_shape=input_shape)
            # Quick train
            nn.train(x, y, epochs=epochs, batch_size=32, learning_rate=0.02)
            preds = nn.predict(x)
            mse = float(np.mean((preds.flatten() - y.flatten()) ** 2))
            layers = decode_genome(genome)
            penalty = complexity_penalty(layers, alpha=complexity_alpha)
            return max(0.0, 1.0 / (1.0 + mse) - penalty)
        except Exception:
            return 0.0  # invalid architecture

    return fitness


def genome_to_readable(genome: np.ndarray) -> str:
    """Human-friendly summary of an architecture genome."""
    layers = decode_genome(genome)
    parts = []
    for i, spec in enumerate(layers):
        if spec["type"] == "dense":
            parts.append(f"Dense({spec['size']}, {spec['activation']})")
        else:
            parts.append(f"Dropout({spec.get('rate', 0.2):.2f})")
    return " → ".join(parts) if parts else "Empty"
