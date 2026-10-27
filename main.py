"""
M.A.G.I.C. – Multi-Agent Global Intelligence Collaboration
Entry point & orchestrator.
"""

from __future__ import annotations
import argparse
import sys
import time
import numpy as np

from genetic_algorithm import GeneticAlgorithm
from neural_network import NeuralNetwork
from dynamic_layer import DynamicLayer
from learning_paradigm import LearningParadigm
from memory_module import MemoryModule
from user_interface import UserInterface


def build_system(paradigm: str = "deep_learning") -> dict:
    """Construct the full M.A.G.I.C. stack and return the live objects."""
    memory = MemoryModule(use_persistence=True)
    ui = UserInterface(memory=memory)

    ga = GeneticAlgorithm(
        population_size=40,
        individual_size=64,
        fitness_goal=0.92,
        mutation_rate=0.08,
        crossover_rate=0.75,
        elite_count=3,
        max_generations=80,
        gene_type="real",
    )

    nn = NeuralNetwork(input_shape=(8,), output_shape=1, backend="numpy")
    dl = DynamicLayer(layer_size=32, activation="relu")
    lp = LearningParadigm(paradigm=paradigm)

    # Apply paradigm and seed topology
    nn = lp.apply_paradigm(nn)
    layer_spec = dl.generate()
    nn.add_layer(layer_spec["size"], layer_spec["activation"])
    nn.add_layer(16, "relu")

    return {
        "ga": ga,
        "nn": nn,
        "dl": dl,
        "lp": lp,
        "memory": memory,
        "ui": ui,
    }


def make_fitness_fn(nn: NeuralNetwork, x: np.ndarray, y: np.ndarray):
    """Create a fitness function that evaluates a genome on the current task data."""

    def fitness(genome: np.ndarray) -> float:
        nn.set_weights_from_genome(genome)
        preds = nn.predict(x)
        mse = np.mean((preds.flatten() - y.flatten()) ** 2)
        # Higher is better
        return float(1.0 / (1.0 + mse))

    return fitness


def run_demo(ui: UserInterface, ga: GeneticAlgorithm, nn: NeuralNetwork, memory: MemoryModule):
    """End-to-end demonstration of the evolutionary loop with live progress."""
    print("[M.A.G.I.C.] Generating synthetic regression task …")
    rng = np.random.default_rng(42)
    x = rng.normal(size=(200, 8)).astype(np.float32)
    # Target: non-linear function of the inputs
    y = (np.sin(x[:, 0]) + 0.5 * x[:, 1] ** 2 - 0.3 * x[:, 2] + rng.normal(0, 0.1, 200)).astype(np.float32)
    y = y.reshape(-1, 1)

    fitness_fn = make_fitness_fn(nn, x, y)
    ga.set_evaluate_function(fitness_fn)

    ui.update_progress(status="evolving", message="Genetic evolution started", generation=0)

    print("[M.A.G.I.C.] Starting evolutionary optimisation …")
    result = ga.evolve()

    ui.update_progress(
        status="completed",
        generation=result["generations_run"],
        best_fitness=result["best_fitness"],
        message=f"Evolution finished after {result['generations_run']} generations",
    )

    memory.store_information(
        {
            "event": "evolution_complete",
            "best_fitness": result["best_fitness"],
            "generations": result["generations_run"],
            "history_tail": result["history"][-5:],
        },
        task_signature="demo_regression",
        tags=["evolution", "result"],
    )

    print(f"[M.A.G.I.C.] Best fitness achieved: {result['best_fitness']:.4f}")
    print(f"[M.A.G.I.C.] Generations run: {result['generations_run']}")
    return result


def main():
    parser = argparse.ArgumentParser(description="M.A.G.I.C. – Multi-Agent Global Intelligence Collaboration")
    parser.add_argument("--paradigm", default="deep_learning", help="Initial learning paradigm")
    parser.add_argument("--demo", action="store_true", help="Run a self-contained evolutionary demo")
    parser.add_argument("--ui", action="store_true", help="Launch the user interface (Flask or CLI)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    system = build_system(paradigm=args.paradigm)
    ga, nn, ui, memory = system["ga"], system["nn"], system["ui"], system["memory"]

    print("=" * 60)
    print("  M.A.G.I.C. – Multi-Agent Global Intelligence Collaboration")
    print("  Or4cl3 AI Solutions")
    print("=" * 60)
    print(f"Learning paradigm : {system['lp'].get_paradigm()}")
    print(f"NN backend        : {nn.backend}")
    print(f"GA population     : {ga.population_size}")
    print()

    if args.demo:
        run_demo(ui, ga, nn, memory)

    if args.ui:
        # Run UI in the main thread (blocks)
        ui.run(host=args.host, port=args.port)
    elif not args.demo:
        # Default: quick demo then exit
        run_demo(ui, ga, nn, memory)
        print("\nTip: re-run with --ui to start the interactive interface.")


if __name__ == "__main__":
    main()
