"""
M.A.G.I.C. Core – Genetic Algorithm Engine
Pure-Python + NumPy implementation (DEAP-compatible interface for future upgrade).
"""

from __future__ import annotations
import random
from typing import Callable, List, Optional, Tuple
import numpy as np


class GeneticAlgorithm:
    """
    Adaptive genetic algorithm capable of evolving both binary and real-valued
    individuals. Designed to optimize neural-network topologies and weights.
    """

    def __init__(
        self,
        population_size: int = 50,
        individual_size: int = 20,
        fitness_goal: float = 0.95,
        mutation_rate: float = 0.05,
        crossover_rate: float = 0.7,
        elite_count: int = 2,
        max_generations: int = 200,
        gene_type: str = "binary",  # "binary" | "real"
    ):
        self.population_size = population_size
        self.individual_size = individual_size
        self.fitness_goal = fitness_goal
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = elite_count
        self.max_generations = max_generations
        self.gene_type = gene_type

        self.population: List[np.ndarray] = []
        self.fitness_values: List[float] = []
        self.generation = 0
        self.best_individual: Optional[np.ndarray] = None
        self.best_fitness: float = -np.inf
        self.history: List[dict] = []

        self._evaluate_fn: Callable[[np.ndarray], float] = self._default_evaluate
        self._initialize_population()

    def _initialize_population(self) -> None:
        if self.gene_type == "binary":
            self.population = [
                np.random.randint(0, 2, size=self.individual_size).astype(float)
                for _ in range(self.population_size)
            ]
        else:
            self.population = [
                np.random.uniform(-1.0, 1.0, size=self.individual_size)
                for _ in range(self.population_size)
            ]
        self.fitness_values = [0.0] * self.population_size

    def set_evaluate_function(self, fn: Callable[[np.ndarray], float]) -> None:
        """Inject a custom fitness function (e.g. neural-network accuracy)."""
        self._evaluate_fn = fn

    def _default_evaluate(self, individual: np.ndarray) -> float:
        """Placeholder – maximizes sum of genes (binary case) or RBF-like peak."""
        if self.gene_type == "binary":
            return float(np.sum(individual)) / self.individual_size
        return float(np.exp(-np.sum(individual ** 2)))

    def calculate_fitness(self) -> None:
        self.fitness_values = [self._evaluate_fn(ind) for ind in self.population]
        best_idx = int(np.argmax(self.fitness_values))
        if self.fitness_values[best_idx] > self.best_fitness:
            self.best_fitness = self.fitness_values[best_idx]
            self.best_individual = self.population[best_idx].copy()

    def _tournament_select(self, k: int = 3) -> np.ndarray:
        idxs = random.sample(range(self.population_size), k)
        best = max(idxs, key=lambda i: self.fitness_values[i])
        return self.population[best].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        point = random.randint(1, self.individual_size - 1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mutant = individual.copy()
        for i in range(len(mutant)):
            if random.random() < self.mutation_rate:
                if self.gene_type == "binary":
                    mutant[i] = 1.0 - mutant[i]
                else:
                    mutant[i] += np.random.normal(0, 0.1)
                    mutant[i] = np.clip(mutant[i], -2.0, 2.0)
        return mutant

    def evolve(self, generations: Optional[int] = None) -> dict:
        """
        Run the evolutionary loop.
        Returns a summary dict containing best individual, fitness, and history.
        """
        gens = generations or self.max_generations
        self.calculate_fitness()

        for gen in range(gens):
            self.generation = gen + 1

            # Elitism
            elite_idxs = np.argsort(self.fitness_values)[-self.elite_count :]
            elites = [self.population[i].copy() for i in elite_idxs]

            # Create offspring
            offspring = []
            while len(offspring) < self.population_size - self.elite_count:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                c1, c2 = self._crossover(p1, p2)
                offspring.append(self._mutate(c1))
                if len(offspring) < self.population_size - self.elite_count:
                    offspring.append(self._mutate(c2))

            self.population = elites + offspring[: self.population_size - self.elite_count]
            self.calculate_fitness()

            stats = {
                "generation": self.generation,
                "best_fitness": self.best_fitness,
                "avg_fitness": float(np.mean(self.fitness_values)),
                "std_fitness": float(np.std(self.fitness_values)),
            }
            self.history.append(stats)

            if self.best_fitness >= self.fitness_goal:
                break

        return {
            "best_individual": self.best_individual,
            "best_fitness": self.best_fitness,
            "generations_run": self.generation,
            "history": self.history,
        }

    def get_population(self) -> List[np.ndarray]:
        return [ind.copy() for ind in self.population]
