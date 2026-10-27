"""
M.A.G.I.C. Core – Neural Network
Dual-backend: pure NumPy (always available) + optional TensorFlow/Keras.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Any
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    HAS_TF = True
except ImportError:
    HAS_TF = False


class NeuralNetwork:
    """
    Flexible neural network that can be evolved by the GeneticAlgorithm.
    Supports both a pure-NumPy MLP (default) and a Keras Sequential backend.
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...] = (10,),
        output_shape: int = 1,
        backend: str = "numpy",
    ):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.backend = backend if (backend == "keras" and HAS_TF) else "numpy"
        self.layers: List[dict] = []          # topology description
        self.weights: List[np.ndarray] = []   # for numpy backend
        self.biases: List[np.ndarray] = []
        self.model = None                     # Keras model when used
        self.is_compiled = False
        self.training_history: List[dict] = []

        if self.backend == "keras":
            self.model = models.Sequential()

    def add_layer(self, layer_size: int = 10, activation: str = "relu") -> None:
        """Add a dense layer (works for both backends)."""
        self.layers.append({"size": layer_size, "activation": activation})

        if self.backend == "keras" and self.model is not None:
            if len(self.model.layers) == 0:
                self.model.add(
                    layers.Dense(
                        layer_size,
                        activation=activation,
                        input_shape=self.input_shape,
                    )
                )
            else:
                self.model.add(layers.Dense(layer_size, activation=activation))
        else:
            # NumPy path – weights will be initialised on first forward / train
            pass

    def _init_numpy_weights(self) -> None:
        if self.weights:
            return
        prev = self.input_shape[0]
        for layer in self.layers:
            size = layer["size"]
            # Xavier-like init
            limit = np.sqrt(6.0 / (prev + size))
            self.weights.append(np.random.uniform(-limit, limit, (prev, size)))
            self.biases.append(np.zeros(size))
            prev = size
        # Output layer if not already present
        if not self.layers or self.layers[-1]["size"] != self.output_shape:
            limit = np.sqrt(6.0 / (prev + self.output_shape))
            self.weights.append(np.random.uniform(-limit, limit, (prev, self.output_shape)))
            self.biases.append(np.zeros(self.output_shape))

    def _activate(self, x: np.ndarray, name: str) -> np.ndarray:
        if name == "relu":
            return np.maximum(0, x)
        if name == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        if name == "tanh":
            return np.tanh(x)
        return x  # linear

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.backend == "keras" and self.model is not None:
            return self.model.predict(x, verbose=0)

        self._init_numpy_weights()
        a = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            act = self.layers[i]["activation"] if i < len(self.layers) else "linear"
            a = self._activate(z, act)
        return a

    def compile_model(self, optimizer: str = "adam", loss: str = "mse") -> None:
        if self.backend == "keras" and self.model is not None:
            self.model.compile(optimizer=optimizer, loss=loss)
        self.is_compiled = True

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.01,
    ) -> dict:
        """Train the network. Returns simple metrics."""
        if self.backend == "keras" and self.model is not None:
            if not self.is_compiled:
                self.compile_model()
            history = self.model.fit(
                x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0
            )
            metrics = {
                "loss": float(history.history["loss"][-1]),
                "epochs": epochs,
            }
            self.training_history.append(metrics)
            return metrics

        # Pure NumPy SGD (very basic – sufficient for evolutionary co-training)
        self._init_numpy_weights()
        n = x_train.shape[0]
        losses = []
        for epoch in range(epochs):
            # mini-batch
            idx = np.random.permutation(n)[:batch_size]
            xb, yb = x_train[idx], y_train[idx]

            # forward
            activations = [xb]
            a = xb
            for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                z = a @ w + b
                act_name = self.layers[i]["activation"] if i < len(self.layers) else "linear"
                a = self._activate(z, act_name)
                activations.append(a)

            # MSE loss
            loss = np.mean((a - yb) ** 2)
            losses.append(loss)

            # backward (simplified)
            delta = 2 * (a - yb) / batch_size
            for i in reversed(range(len(self.weights))):
                a_prev = activations[i]
                grad_w = a_prev.T @ delta
                grad_b = np.sum(delta, axis=0)
                self.weights[i] -= learning_rate * grad_w
                self.biases[i] -= learning_rate * grad_b
                if i > 0:
                    delta = (delta @ self.weights[i].T) * (activations[i] > 0)  # ReLU-ish

        metrics = {"loss": float(np.mean(losses[-5:])), "epochs": epochs}
        self.training_history.append(metrics)
        return metrics

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def get_topology(self) -> List[dict]:
        return list(self.layers)

    def set_weights_from_genome(self, genome: np.ndarray) -> None:
        """Decode a flat genome into the network weights (for GA evolution)."""
        self._init_numpy_weights()
        pointer = 0
        new_weights = []
        new_biases = []
        for w, b in zip(self.weights, self.biases):
            w_size = w.size
            b_size = b.size
            if pointer + w_size + b_size > len(genome):
                break
            new_w = genome[pointer : pointer + w_size].reshape(w.shape)
            pointer += w_size
            new_b = genome[pointer : pointer + b_size]
            pointer += b_size
            new_weights.append(new_w)
            new_biases.append(new_b)
        if new_weights:
            self.weights = new_weights
            self.biases = new_biases
