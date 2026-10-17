"""Optimiseur : descente de gradient (SGD)."""


class SGD:
    def __init__(self, lr: float = 0.1):
        self.lr = lr

    def step(self, model, grads: dict) -> None:
        """Met à jour les paramètres du modèle : param <- param - lr * gradient."""
        model.W = model.W - self.lr * grads["W"]
        model.b = model.b - self.lr * grads["b"]
