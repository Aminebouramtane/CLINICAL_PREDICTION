"""ClinicalPredictor : diagnostic à partir d'un modèle DÉJÀ entraîné.

Duck typing : tout objet possédant une méthode predict() peut être utilisé.
"""
import numpy as np


class ClinicalPredictor:
    def __init__(self, trained_model, threshold: float = 0.5):
        if not hasattr(trained_model, "predict"):
            raise TypeError("Le modèle doit posséder une méthode predict()")
        self.model = trained_model
        self.threshold = threshold

    def diagnose(self, patient_data) -> str:
        """Retourne "Infecté" si prédiction >= 0.5, "Sain" sinon."""
        x = np.atleast_2d(np.asarray(patient_data, dtype=float))
        prediction = float(np.ravel(self.model.predict(x))[0])
        return "Infecté" if prediction >= self.threshold else "Sain"
