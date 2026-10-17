"""Dataset : chargement des données patients depuis un fichier CSV."""
import csv

import numpy as np


class Dataset:
    def __init__(self, path: str, target: str = "infecte"):
        self.path = path
        self.target = target
        self.feature_names = []
        self.X = None
        self.y = None

    def load(self) -> "Dataset":
        with open(self.path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.feature_names = [c for c in rows[0] if c != self.target]
        self.X = np.array([[float(r[c]) for c in self.feature_names] for r in rows])
        self.y = np.array([int(r[self.target]) for r in rows])
        return self

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def __len__(self) -> int:
        return len(self.y)

    def __repr__(self) -> str:
        return f"Dataset(n={len(self)}, features={self.feature_names})"
