"""Trainer : entraîne un modèle (séparé de la prédiction)."""
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, epochs: int = 100, batch_size: int = 32, seed: int = 0):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.history = []

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> list:
        n = len(y)
        for epoch in range(1, self.epochs + 1):
            idx = self.rng.permutation(n)
            for start in range(0, n, self.batch_size):
                b = idx[start:start + self.batch_size]
                self.optimizer.step(self.model, self.model.gradients(X[b], y[b]))
            self.history.append(self.model.loss(X, y))
            if verbose and (epoch == 1 or epoch % 20 == 0):
                print(f"epoch {epoch:3d}/{self.epochs}  loss={self.history[-1]:.4f}")
        return self.history
