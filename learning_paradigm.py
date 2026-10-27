"""
M.A.G.I.C. Core – Learning Paradigm Switcher
Allows the system to change its learning strategy at runtime.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional


class LearningParadigm:
    """
    Manages the active learning paradigm and applies paradigm-specific
    modifications to a neural network or training loop.
    """

    SUPPORTED = {
        "deep_learning",
        "reinforcement_learning",
        "transfer_learning",
        "self_supervised_learning",
        "self_attention_mechanisms",
        "self_reflection",
        "ensemble_learning",
        "unsupervised_learning",
        "meta_learning",
    }

    def __init__(self, paradigm: str = "deep_learning"):
        if paradigm not in self.SUPPORTED:
            raise ValueError(f"Unknown paradigm '{paradigm}'. Supported: {self.SUPPORTED}")
        self.paradigm = paradigm
        self._handlers: Dict[str, Callable] = {
            "deep_learning": self._apply_deep,
            "reinforcement_learning": self._apply_rl,
            "transfer_learning": self._apply_transfer,
            "self_supervised_learning": self._apply_ssl,
            "self_attention_mechanisms": self._apply_attention,
            "self_reflection": self._apply_reflection,
            "ensemble_learning": self._apply_ensemble,
            "unsupervised_learning": self._apply_unsupervised,
            "meta_learning": self._apply_meta,
        }

    def switch_paradigm(self, new_paradigm: str) -> None:
        if new_paradigm not in self.SUPPORTED:
            raise ValueError(f"Unknown paradigm '{new_paradigm}'")
        self.paradigm = new_paradigm

    def get_paradigm(self) -> str:
        return self.paradigm

    def apply_paradigm(self, network: Any) -> Any:
        """
        Apply the current paradigm to the given neural network object.
        Returns the (possibly modified) network.
        """
        handler = self._handlers.get(self.paradigm, self._apply_deep)
        return handler(network)

    # ----- paradigm-specific handlers -----

    def _apply_deep(self, network: Any) -> Any:
        """Standard supervised deep learning – ensure at least one hidden layer."""
        if hasattr(network, "layers") and len(getattr(network, "layers", [])) == 0:
            if hasattr(network, "add_layer"):
                network.add_layer(64, "relu")
                network.add_layer(32, "relu")
        return network

    def _apply_rl(self, network: Any) -> Any:
        """Reinforcement-learning flavour – prefer smaller nets + linear output."""
        if hasattr(network, "add_layer"):
            network.add_layer(32, "relu")
            network.add_layer(16, "tanh")
        return network

    def _apply_transfer(self, network: Any) -> Any:
        """Transfer learning placeholder – mark network as 'frozen backbone'."""
        if hasattr(network, "layers"):
            # In a full implementation we would freeze early layers
            pass
        return network

    def _apply_ssl(self, network: Any) -> Any:
        """Self-supervised – typically needs a projection head."""
        if hasattr(network, "add_layer"):
            network.add_layer(128, "relu")  # projection
        return network

    def _apply_attention(self, network: Any) -> Any:
        """Self-attention – for now just add a wider layer as proxy."""
        if hasattr(network, "add_layer"):
            network.add_layer(64, "tanh")
        return network

    def _apply_reflection(self, network: Any) -> Any:
        """Self-reflection – meta-cognitive loop (placeholder)."""
        return network

    def _apply_ensemble(self, network: Any) -> Any:
        """Ensemble – flag that multiple models should be trained."""
        if hasattr(network, "__dict__"):
            network._ensemble = True
        return network

    def _apply_unsupervised(self, network: Any) -> Any:
        """Unsupervised – auto-encoder style bottleneck."""
        if hasattr(network, "add_layer"):
            network.add_layer(16, "relu")  # bottleneck
            network.add_layer(32, "relu")
        return network

    def _apply_meta(self, network: Any) -> Any:
        """Meta-learning – ready for MAML-style outer loop."""
        return network
