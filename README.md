# Clinical Prediction

Petit projet de diagnostic clinique orienté objet en Python. Un modèle de **régression logistique** est entraîné sur des données de patients, puis utilisé par une classe `ClinicalPredictor` qui renvoie **"Infecté"** ou **"Sain"**, via une interface graphique Tkinter.

> Exercice d'application du Chap. 5 : Programmation orientée objet.

## Structure du projet

```
clinical_prediction/
├── core/
│   ├── __init__.py
│   ├── dataset.py              # Dataset : chargement du CSV
│   ├── logistic_regression.py  # LogisticRegression : modèle
│   └── optimizer.py            # SGD : descente de gradient
├── pipeline/
│   ├── __init__.py
│   └── trainer.py              # Trainer : entraînement du modèle
├── utils/
│   ├── __init__.py
│   └── preprocessing.py        # StandardScaler, train_test_split
├── app/
│   ├── __init__.py
│   ├── clinical_predictor.py   # ClinicalPredictor : diagnostic
│   └── interface.py            # ClinicalApp : interface Tkinter
├── data/
│   └── patient_data.csv        # Données patients (synthétiques)
└── main.py                     # Point d'entrée
```

## Prérequis

- Python 3.8 ou plus récent
- `numpy`
- `tkinter` (inclus avec Python sous Windows et macOS ; sous Linux : `sudo apt install python3-tk`)

## Installation et lancement

```bash
cd clinical_prediction
pip install numpy
python main.py
```

L'entraînement s'affiche dans le terminal, puis la fenêtre de diagnostic s'ouvre.

## Utilisation

Dans la fenêtre, renseigne la température, l'âge, la saturation en O2, coche les symptômes (toux, fatigue, contact avec un infecté) et clique sur **Diagnostiquer**. Le résultat s'affiche en rouge (Infecté) ou en vert (Sain), avec la probabilité d'infection.

Utilisation en code :

```python
from app.clinical_predictor import ClinicalPredictor

predictor = ClinicalPredictor(trained_model)   # modèle déjà entraîné
predictor.diagnose(patient_data)               # "Infecté" ou "Sain"
```

## Données

`data/patient_data.csv` contient 300 patients **synthétiques** avec les colonnes :

| Colonne | Description |
|---|---|
| `temperature` | Température corporelle (°C) |
| `toux` | 1 = oui, 0 = non |
| `fatigue` | 1 = oui, 0 = non |
| `age` | Âge (années) |
| `saturation_o2` | Saturation en oxygène (%) |
| `contact_infecte` | 1 = contact avec une personne infectée |
| `infecte` | Cible : 1 = infecté, 0 = sain |

Les données sont fictives. Les résultats obtenus (proches de 100 % de précision) ne sont pas représentatifs d'un cas réel.

## Concepts de POO illustrés

- **Encapsulation** : chaque classe a une responsabilité (`Dataset`, `Trainer`, `ClinicalPredictor`).
- **Polymorphisme / duck typing** : `ClinicalPredictor` fonctionne avec n'importe quel modèle possédant une méthode `predict()`.
- **Séparation entraînement / prédiction** : le `Trainer` apprend, le `ClinicalPredictor` diagnostique avec un modèle figé.

