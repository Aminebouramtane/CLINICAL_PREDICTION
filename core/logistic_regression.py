"""Régression logistique : calcule P(infecté | patient)."""
import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegression:
    def __init__(self, n_features: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, size=n_features)
        self.b = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Retourne une probabilité entre 0 et 1 pour chaque patient."""
        return sigmoid(np.atleast_2d(X) @ self.W + self.b)

    def gradients(self, X: np.ndarray, y: np.ndarray) -> dict:
        err = self.predict(X) - y
        return {"W": X.T @ err / len(y), "b": float(np.mean(err))}

    def loss(self, X: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
        p = np.clip(self.predict(X), eps, 1 - eps)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
