"""Point d'entrée : charge, entraîne, évalue, puis ouvre l'interface graphique."""
from pathlib import Path

import numpy as np

from app.clinical_predictor import ClinicalPredictor
from app.interface import ClinicalApp
from core.dataset import Dataset
from core.logistic_regression import LogisticRegression
from core.optimizer import SGD
from pipeline.trainer import Trainer
from utils.preprocessing import StandardScaler, train_test_split


def main() -> None:
    # 1. Données
    ds = Dataset(str(Path(__file__).parent / "data" / "patient_data.csv")).load()
    print(ds)
    X_tr, X_te, y_tr, y_te = train_test_split(ds.X, ds.y)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # 2. Entraînement (Trainer)
    model = LogisticRegression(ds.n_features)
    Trainer(model, SGD(lr=0.1), epochs=100).fit(X_tr, y_tr)

    # 3. Évaluation rapide sur le jeu de test
    acc = np.mean((model.predict(X_te) >= 0.5).astype(int) == y_te)
    print(f"Accuracy (test) : {acc:.3f}")

    # 4. Interface graphique avec le modèle déjà entraîné
    predictor = ClinicalPredictor(model)
    ClinicalApp(predictor, scaler).mainloop()


if __name__ == "__main__":
    main()
