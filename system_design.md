# M.A.G.I.C. System Design – v0.2

## Implementation Approach (Updated)

M.A.G.I.C. is implemented as a pure-Python package with a minimal core dependency on NumPy. Optional backends (DEAP, TensorFlow/Keras, Flask) can be enabled when available. The design prioritises:

- Zero hard external dependencies for the evolutionary core
- Clear public APIs that match the original 2023 class diagram
- Injectable fitness functions so the same GA can optimise any task
- Persistent experience storage for continual learning
- Live telemetry so a monitoring UI can be attached without invasive changes

## Package Layout

```
magic2/
├── genetic_algorithm.py   # Adaptive GA (binary + real-valued)
├── neural_network.py      # Dual-backend NN (NumPy / Keras)
├── dynamic_layer.py       # Layer specification generator + mutation
├── learning_paradigm.py   # Runtime paradigm switching
├── memory_module.py       # SQLite-backed experience store
├── user_interface.py      # Flask API + CLI fallback
├── main.py                # Orchestrator + demo
├── __init__.py            # Clean public exports
├── requirements.txt
├── prd.md                 # Original product requirements
└── system_design.md       # This document
```

## Class Responsibilities (v0.2)

- **GeneticAlgorithm** – population management, selection, crossover, mutation, elitism, history, injectable fitness.
- **NeuralNetwork** – topology description, forward pass, training (NumPy SGD or Keras), genome-to-weight decoding.
- **DynamicLayer** – produces serialisable layer specs that the GA can mutate.
- **LearningParadigm** – switches learning strategy and applies paradigm-specific topology modifications.
- **MemoryModule** – stores and retrieves experiences by task signature and tags.
- **UserInterface** – task submission, progress monitoring, feedback collection; works with or without Flask.

## Data Flow

1. User (or demo) submits a task via UserInterface.
2. MemoryModule records the task.
3. GeneticAlgorithm is configured with a fitness function that evaluates candidate genomes on the task (via NeuralNetwork).
4. Evolution runs; progress is pushed live to UserInterface.
5. Best individual and summary metrics are stored in MemoryModule.
6. User can supply feedback, which is also persisted.

## Maturity Assessment

| Concern | v0.1 (2023) | v0.2 (2026) |
|---------|-------------|-------------|
| Runnable out-of-the-box | No | Yes |
| Fitness injection | Placeholder | Supported |
| Termination safety | Infinite-loop risk | Generation limit + goal |
| Memory durability | In-memory list | SQLite |
| UI | Hard-coded Flask | Flask + CLI fallback |
| Paradigm handlers | Empty `pass` | Functional topology changes |
| Packaging | Flat scripts | Proper package with `__init__` |

## Roadmap Alignment

- **P0** (minimalist base + UI) → Complete
- **P1** (dynamic layers, paradigm integration, continual learning) → Core mechanisms in place
- **P2** (adaptive GA for diverse networks, rich memory) → Foundation ready; deeper architecture search and multi-agent coordination are the natural next increments

## Anything UNCLEAR

None. The original vision remains the north star; v0.2 simply makes that vision executable.
