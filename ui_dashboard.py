"""
M.A.G.I.C. – Modern UI Dashboard (Streamlit)
Run with:  streamlit run ui_dashboard.py
"""

from __future__ import annotations
import time
import json
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st

# Core M.A.G.I.C. imports
from genetic_algorithm import GeneticAlgorithm
from neural_network import NeuralNetwork
from dynamic_layer import DynamicLayer
from learning_paradigm import LearningParadigm
from memory_module import MemoryModule
from architecture_genome import (
    GENOME_LENGTH,
    architecture_fitness_factory,
    genome_to_readable,
    decode_genome,
)

from agents.message_bus import MessageBus
from agents.coordinator import Coordinator
from agents.specialist_agents import (
    ExplorerAgent,
    OptimizerAgent,
    CriticAgent,
    MemoryKeeperAgent,
)


# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="M.A.G.I.C. Dashboard",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700&family=Inter:wght@400;500;600&display=swap');

    .stApp {
        background: linear-gradient(160deg, #03040a 0%, #0a0c18 40%, #0d1024 100%);
        color: #e8ecff;
    }
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        letter-spacing: 0.03em;
    }
    .main-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6c8cff, #a78bfa, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.25rem;
    }
    .sub-header {
        color: #8b93b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: rgba(16, 20, 36, 0.75);
        border: 1px solid rgba(120, 140, 255, 0.18);
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #6c8cff;
    }
    .metric-label {
        font-size: 0.78rem;
        color: #8b93b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.25rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6c8cff, #a78bfa) !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1.4rem !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 24px rgba(108, 140, 255, 0.45) !important;
    }
    div[data-testid="stSidebar"] {
        background: rgba(8, 10, 20, 0.95);
        border-right: 1px solid rgba(120, 140, 255, 0.12);
    }
    .agent-badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.15rem;
        background: rgba(108, 140, 255, 0.15);
        border: 1px solid rgba(108, 140, 255, 0.3);
        color: #a5b4fc;
    }
    .log-box {
        background: #07090f;
        border: 1px solid rgba(120, 140, 255, 0.15);
        border-radius: 12px;
        padding: 1rem;
        font-family: ui-monospace, monospace;
        font-size: 0.82rem;
        max-height: 280px;
        overflow-y: auto;
        color: #c8d0f0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------
def init_state():
    if "system" not in st.session_state:
        st.session_state.system = None
    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "last_result" not in st.session_state:
        st.session_state.last_result = None
    if "running" not in st.session_state:
        st.session_state.running = False


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.logs.append(f"[{ts}] {msg}")
    if len(st.session_state.logs) > 80:
        st.session_state.logs = st.session_state.logs[-80:]


def build_magic_system(paradigm: str = "deep_learning") -> dict:
    memory = MemoryModule(use_persistence=True, db_path="magic_ui_memory.db")
    bus = MessageBus()

    ga = GeneticAlgorithm(
        population_size=32,
        individual_size=64,
        fitness_goal=0.95,
        mutation_rate=0.08,
        crossover_rate=0.75,
        elite_count=3,
        max_generations=60,
        gene_type="real",
    )

    nn = NeuralNetwork(input_shape=(8,), output_shape=1, backend="numpy")
    dl = DynamicLayer(layer_size=32, activation="relu")
    lp = LearningParadigm(paradigm=paradigm)
    nn = lp.apply_paradigm(nn)
    spec = dl.generate()
    nn.add_layer(spec["size"], spec["activation"])
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

    coordinator = Coordinator(bus=bus, memory=memory, ui=None)

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
        "bus": bus,
        "coordinator": coordinator,
        "agents": {
            "explorer": explorer,
            "optimizer": optimizer,
            "critic": critic,
            "memory_keeper": memory_keeper,
        },
        "data": (x, y),
    }


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------
def sidebar():
    with st.sidebar:
        st.markdown("### ✨ M.A.G.I.C. Control")
        st.caption("Multi-Agent Global Intelligence Collaboration")

        paradigm = st.selectbox(
            "Learning Paradigm",
            [
                "deep_learning",
                "reinforcement_learning",
                "meta_learning",
                "ensemble_learning",
                "self_supervised_learning",
                "transfer_learning",
            ],
            index=0,
        )

        st.divider()
        st.markdown("**Population Settings**")
        pop_size = st.slider("Population size", 10, 80, 32, 2)
        max_gen = st.slider("Max generations", 5, 100, 25, 5)
        mut_rate = st.slider("Mutation rate", 0.01, 0.3, 0.08, 0.01)

        st.divider()
        if st.button("🔄 Initialize / Reset System", use_container_width=True):
            with st.spinner("Building M.A.G.I.C. stack..."):
                system = build_magic_system(paradigm=paradigm)
                system["ga"].population_size = pop_size
                system["ga"].max_generations = max_gen
                system["ga"].mutation_rate = mut_rate
                system["ga"]._initialize_population()
                st.session_state.system = system
                st.session_state.logs = []
                st.session_state.last_result = None
                log(f"System initialized · paradigm={paradigm} · pop={pop_size}")
            st.success("System ready")

        st.divider()
        st.markdown("**Quick Actions**")
        col1, col2 = st.columns(2)
        with col1:
            run_classic = st.button("Classic Evolve", use_container_width=True)
        with col2:
            run_arch = st.button("Arch Search", use_container_width=True)
        run_multi = st.button("🤖 Multi-Agent Pipeline", use_container_width=True)

        return {
            "paradigm": paradigm,
            "pop_size": pop_size,
            "max_gen": max_gen,
            "mut_rate": mut_rate,
            "run_classic": run_classic,
            "run_arch": run_arch,
            "run_multi": run_multi,
        }


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------
def run_classic_evolution(system: dict, generations: int):
    ga = system["ga"]
    log(f"Classic weight evolution started · {generations} gens")
    progress = st.progress(0.0, text="Evolving weights...")
    status = st.empty()

    ga.calculate_fitness()
    history = []
    for gen in range(generations):
        ga.generation = gen + 1
        elite_idxs = np.argsort(ga.fitness_values)[-ga.elite_count:]
        elites = [ga.population[i].copy() for i in elite_idxs]
        offspring = []
        while len(offspring) < ga.population_size - ga.elite_count:
            p1 = ga._tournament_select()
            p2 = ga._tournament_select()
            c1, c2 = ga._crossover(p1, p2)
            offspring.append(ga._mutate(c1))
            if len(offspring) < ga.population_size - ga.elite_count:
                offspring.append(ga._mutate(c2))
        ga.population = elites + offspring[: ga.population_size - ga.elite_count]
        ga.calculate_fitness()
        stats = {
            "generation": ga.generation,
            "best_fitness": ga.best_fitness,
            "avg_fitness": float(np.mean(ga.fitness_values)),
        }
        history.append(stats)
        progress.progress((gen + 1) / generations, text=f"Gen {gen+1}/{generations} · best={ga.best_fitness:.4f}")
        status.caption(f"Best fitness: **{ga.best_fitness:.4f}**  |  Avg: {stats['avg_fitness']:.4f}")
        if ga.best_fitness >= ga.fitness_goal:
            break

    result = {
        "best_fitness": ga.best_fitness,
        "generations_run": ga.generation,
        "history": history,
        "mode": "weights",
    }
    system["memory"].store_information(
        {"event": "classic_evolution", "result": result},
        task_signature="ui_classic",
        tags=["ui", "classic"],
    )
    log(f"Classic evolution finished · fitness={ga.best_fitness:.4f} · gens={ga.generation}")
    return result


def run_architecture_search(system: dict, generations: int):
    ga = system["ga"]
    x, y = system["data"]
    log(f"Architecture search started · {generations} gens")

    ga.individual_size = GENOME_LENGTH
    ga.gene_type = "real"
    ga.max_generations = generations
    ga._initialize_population()
    ga.set_evaluate_function(architecture_fitness_factory(x, y, epochs=3, complexity_alpha=0.001))

    progress = st.progress(0.0, text="Searching architectures...")
    status = st.empty()

    ga.calculate_fitness()
    history = []
    for gen in range(generations):
        ga.generation = gen + 1
        elite_idxs = np.argsort(ga.fitness_values)[-ga.elite_count:]
        elites = [ga.population[i].copy() for i in elite_idxs]
        offspring = []
        while len(offspring) < ga.population_size - ga.elite_count:
            p1 = ga._tournament_select()
            p2 = ga._tournament_select()
            c1, c2 = ga._crossover(p1, p2)
            offspring.append(ga._mutate(c1))
            if len(offspring) < ga.population_size - ga.elite_count:
                offspring.append(ga._mutate(c2))
        ga.population = elites + offspring[: ga.population_size - ga.elite_count]
        ga.calculate_fitness()
        stats = {
            "generation": ga.generation,
            "best_fitness": ga.best_fitness,
            "avg_fitness": float(np.mean(ga.fitness_values)),
        }
        history.append(stats)
        arch = genome_to_readable(ga.best_individual) if ga.best_individual is not None else "—"
        progress.progress((gen + 1) / generations, text=f"Gen {gen+1}/{generations} · best={ga.best_fitness:.4f}")
        status.caption(f"Best: **{ga.best_fitness:.4f}**  ·  `{arch}`")

    arch = genome_to_readable(ga.best_individual) if ga.best_individual is not None else "n/a"
    result = {
        "best_fitness": ga.best_fitness,
        "generations_run": ga.generation,
        "history": history,
        "architecture": arch,
        "layers": decode_genome(ga.best_individual) if ga.best_individual is not None else [],
        "mode": "architecture",
    }
    system["memory"].store_information(
        {"event": "architecture_search", "architecture": arch, "fitness": ga.best_fitness},
        task_signature="ui_arch",
        tags=["ui", "architecture"],
    )
    log(f"Architecture search finished · fitness={ga.best_fitness:.4f} · arch={arch}")
    return result


def run_multi_agent(system: dict, generations: int = 20):
    coordinator = system["coordinator"]
    log("Multi-agent pipeline started")
    goal = "Evolve a compact model for the synthetic regression task and critique the result"
    with st.spinner("Agents negotiating and executing..."):
        goal_id = coordinator.submit_goal(goal, context={"generations": generations, "mode": "weights"})
        report = coordinator.status_report()
    log(f"Multi-agent goal complete · id={goal_id[:8]} · sub-tasks={report['completed']}")
    return {"goal_id": goal_id, "report": report}


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------
def main():
    init_state()
    controls = sidebar()

    st.markdown('<div class="main-header">M.A.G.I.C.</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Multi-Agent Global Intelligence Collaboration · Live Dashboard</div>',
        unsafe_allow_html=True,
    )

    system = st.session_state.system

    # ---- Top metrics ----
    m1, m2, m3, m4 = st.columns(4)
    if system:
        ga = system["ga"]
        with m1:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{ga.population_size}</div>'
                f'<div class="metric-label">Population</div></div>',
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{ga.generation}</div>'
                f'<div class="metric-label">Generation</div></div>',
                unsafe_allow_html=True,
            )
        with m3:
            best = f"{ga.best_fitness:.3f}" if ga.best_fitness > -np.inf else "—"
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{best}</div>'
                f'<div class="metric-label">Best Fitness</div></div>',
                unsafe_allow_html=True,
            )
        with m4:
            n_agents = len(system["agents"])
            st.markdown(
                f'<div class="metric-card"><div class="metric-value">{n_agents}</div>'
                f'<div class="metric-label">Agents Online</div></div>',
                unsafe_allow_html=True,
            )
    else:
        for col in (m1, m2, m3, m4):
            with col:
                st.markdown(
                    '<div class="metric-card"><div class="metric-value">—</div>'
                    '<div class="metric-label">Initialize system</div></div>',
                    unsafe_allow_html=True,
                )

    st.markdown("")

    # ---- Action triggers ----
    if system is None:
        st.info("← Use the sidebar to **Initialize / Reset System**, then run a demo.")
    else:
        if controls["run_classic"]:
            st.session_state.running = True
            result = run_classic_evolution(system, controls["max_gen"])
            st.session_state.last_result = result
            st.session_state.running = False
            st.rerun()

        if controls["run_arch"]:
            st.session_state.running = True
            result = run_architecture_search(system, min(controls["max_gen"], 20))
            st.session_state.last_result = result
            st.session_state.running = False
            st.rerun()

        if controls["run_multi"]:
            st.session_state.running = True
            result = run_multi_agent(system, generations=min(controls["max_gen"], 25))
            st.session_state.last_result = result
            st.session_state.running = False
            st.rerun()

    # ---- Main panels ----
    left, right = st.columns([1.35, 1])

    with left:
        st.subheader("Latest Result")
        result = st.session_state.last_result
        if result is None:
            st.caption("No runs yet. Launch a demo from the sidebar.")
        else:
            if "architecture" in result:
                st.success(f"**Architecture found:** `{result['architecture']}`")
                st.metric("Best Fitness", f"{result['best_fitness']:.4f}")
                st.metric("Generations", result["generations_run"])
                if result.get("layers"):
                    st.json(result["layers"])
            elif "goal_id" in result:
                st.success(f"**Multi-agent goal complete**")
                st.code(result["goal_id"], language=None)
                report = result.get("report", {})
                st.write("**Agents**")
                for name, info in report.get("agents", {}).items():
                    st.markdown(
                        f'<span class="agent-badge">{name} · {info.get("status", "?")} · load {info.get("load", 0):.1f}</span>',
                        unsafe_allow_html=True,
                    )
                st.metric("Sub-tasks completed", report.get("completed", 0))
            else:
                st.metric("Best Fitness", f"{result.get('best_fitness', 0):.4f}")
                st.metric("Generations", result.get("generations_run", 0))
                if result.get("history"):
                    import pandas as pd
                    df = pd.DataFrame(result["history"])
                    st.line_chart(df.set_index("generation")[["best_fitness", "avg_fitness"]])

        st.subheader("Activity Log")
        log_html = "<br>".join(st.session_state.logs[-30:][::-1]) or "No activity yet."
        st.markdown(f'<div class="log-box">{log_html}</div>', unsafe_allow_html=True)

    with right:
        st.subheader("Agents")
        if system:
            for name, agent in system["agents"].items():
                status_color = "#22d3ee" if agent.status == "idle" else "#fbbf24"
                st.markdown(
                    f"""
                    <div style="background:rgba(16,20,36,0.7);border:1px solid rgba(120,140,255,0.15);
                                border-radius:12px;padding:0.85rem 1rem;margin-bottom:0.6rem;">
                        <strong style="font-family:Orbitron,sans-serif;">{name}</strong>
                        <span style="float:right;color:{status_color};font-size:0.8rem;">{agent.status}</span><br>
                        <span style="color:#8b93b8;font-size:0.78rem;">
                            role: {agent.role} · load: {agent.load:.2f}<br>
                            caps: {', '.join(sorted(agent.capabilities))}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Initialize the system to see agents.")

        st.subheader("Memory")
        if system:
            entries = system["memory"].retrieve_information(limit=8)
            if not entries:
                st.caption("Memory is empty.")
            else:
                for e in reversed(entries):
                    content = e.get("content", {})
                    event = content.get("event", "entry") if isinstance(content, dict) else str(content)[:60]
                    sig = e.get("task_signature", "—")
                    st.markdown(f"`{sig[:12]}` · **{event}**")
        else:
            st.caption("Initialize the system to browse memory.")

    st.divider()
    st.caption("M.A.G.I.C. v0.4 · Or4cl3 AI Solutions · Built with Streamlit")


if __name__ == "__main__":
    main()
