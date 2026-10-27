"""
M.A.G.I.C. – Multi-Agent Global Intelligence Collaboration
Entry point & orchestrator (v0.3 – multi-agent layer online).
"""

from __future__ import annotations
import argparse
import numpy as np

from genetic_algorithm import GeneticAlgorithm
from neural_network import NeuralNetwork
from dynamic_layer import DynamicLayer
from learning_paradigm import LearningParadigm
from memory_module import MemoryModule
from user_interface import UserInterface

from agents.message_bus import MessageBus
from agents.coordinator import Coordinator
from agents.specialist_agents import (
    ExplorerAgent,
    OptimizerAgent,
    CriticAgent,
    MemoryKeeperAgent,
)


def build_system(paradigm: str = "deep_learning") -> dict:
    """Construct the full M.A.G.I.C. stack including the multi-agent layer."""
    memory = MemoryModule(use_persistence=True)
    ui = UserInterface(memory=memory)
    bus = MessageBus()

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

    nn = lp.apply_paradigm(nn)
    layer_spec = dl.generate()
    nn.add_layer(layer_spec["size"], layer_spec["activation"])
    nn.add_layer(16, "relu")

    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 8)).astype(np.float32)
    y = (np.sin(x[:, 0]) + 0.4 * x[:, 1] ** 2 + rng.normal(0, 0.08, 120)).astype(np.float32).reshape(-1, 1)

    def fitness(genome):
        nn.set_weights_from_genome(genome)
        preds = nn.predict(x)
        mse = float(np.mean((preds.flatten() - y.flatten()) ** 2))
        return 1.0 / (1.0 + mse)

    ga.set_evaluate_function(fitness)

    coordinator = Coordinator(bus=bus, memory=memory, ui=ui)

    explorer = ExplorerAgent(bus)
    optimizer = OptimizerAgent(bus, ga=ga, nn=nn)
    critic = CriticAgent(bus)
    memory_keeper = MemoryKeeperAgent(bus, memory=memory)

    for agent in (explorer, optimizer, critic, memory_keeper):
        coordinator.register_agent(agent)

    return {
        "ga": ga,
        "nn": nn,
        "dl": dl,
        "lp": lp,
        "memory": memory,
        "ui": ui,
        "bus": bus,
        "coordinator": coordinator,
        "agents": {
            "explorer": explorer,
            "optimizer": optimizer,
            "critic": critic,
            "memory_keeper": memory_keeper,
        },
    }


def run_classic_demo(ui, ga, nn, memory):
    print("[M.A.G.I.C.] Classic evolutionary demo …")
    result = ga.evolve()
    ui.update_progress(
        status="completed",
        generation=result["generations_run"],
        best_fitness=result["best_fitness"],
        message=f"Classic evolution finished – fitness {result['best_fitness']:.4f}",
    )
    memory.store_information(
        {"event": "classic_evolution", "result": result},
        task_signature="classic_demo",
        tags=["demo", "classic"],
    )
    print(f"[M.A.G.I.C.] Best fitness: {result['best_fitness']:.4f}  generations: {result['generations_run']}")
    return result


def run_multi_agent_demo(coordinator, ui):
    print("[M.A.G.I.C.] Multi-agent collaboration demo …")
    print("             Agents: Explorer → Optimizer → Critic → MemoryKeeper")
    goal = "Evolve a compact regressor for a non-linear synthetic target and critique the result"
    goal_id = coordinator.submit_goal(goal, context={"generations": 35})
    report = coordinator.status_report()
    print(f"[M.A.G.I.C.] Goal ID     : {goal_id}")
    print(f"[M.A.G.I.C.] Agents      : {list(report['agents'].keys())}")
    print(f"[M.A.G.I.C.] Completed   : {report['completed']} sub-tasks")
    ui.update_progress(status="multi_agent_complete", message="Multi-agent pipeline finished")
    return goal_id


def main():
    parser = argparse.ArgumentParser(description="M.A.G.I.C. – Multi-Agent Global Intelligence Collaboration")
    parser.add_argument("--paradigm", default="deep_learning", help="Initial learning paradigm")
    parser.add_argument("--demo", action="store_true", help="Run classic single-process evolutionary demo")
    parser.add_argument("--multi-agent", action="store_true", help="Run multi-agent collaboration demo")
    parser.add_argument("--ui", action="store_true", help="Launch the user interface (Flask or CLI)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    system = build_system(paradigm=args.paradigm)
    ga = system["ga"]
    nn = system["nn"]
    ui = system["ui"]
    memory = system["memory"]
    coordinator = system["coordinator"]

    print("=" * 64)
    print("  M.A.G.I.C. – Multi-Agent Global Intelligence Collaboration")
    print("  Or4cl3 AI Solutions  ·  v0.3")
    print("=" * 64)
    print(f"Learning paradigm : {system['lp'].get_paradigm()}")
    print(f"NN backend        : {nn.backend}")
    print(f"GA population     : {ga.population_size}")
    print(f"Agents online     : {list(system['agents'].keys())}")
    print()

    if args.multi_agent:
        run_multi_agent_demo(coordinator, ui)
    elif args.demo:
        run_classic_demo(ui, ga, nn, memory)
    else:
        run_multi_agent_demo(coordinator, ui)

    if args.ui:
        ui.run(host=args.host, port=args.port)
    elif not (args.demo or args.multi_agent):
        print("\nTip:  python main.py --multi-agent")
        print("      python main.py --demo")
        print("      python main.py --ui")


if __name__ == "__main__":
    main()
