"""Prétraitement : standardisation et découpage train/test."""
import numpy as np


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X) -> np.ndarray:
        return (np.atleast_2d(X) - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def train_test_split(X, y, test_size: float = 0.2, seed: int = 42):
    idx = np.random.default_rng(seed).permutation(len(y))
    n_test = int(len(y) * test_size)
    test, train = idx[:n_test], idx[n_test:]
    return X[train], X[test], y[train], y[test]
