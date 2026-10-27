"""
M.A.G.I.C. – Multi-Agent Global Intelligence Collaboration
Or4cl3 AI Solutions
"""

__version__ = "0.2.0"
__author__ = "Dustin Groves / Or4cl3 AI Solutions"

from .genetic_algorithm import GeneticAlgorithm
from .neural_network import NeuralNetwork
from .dynamic_layer import DynamicLayer
from .learning_paradigm import LearningParadigm
from .memory_module import MemoryModule
from .user_interface import UserInterface

__all__ = [
    "GeneticAlgorithm",
    "NeuralNetwork",
    "DynamicLayer",
    "LearningParadigm",
    "MemoryModule",
    "UserInterface",
]
