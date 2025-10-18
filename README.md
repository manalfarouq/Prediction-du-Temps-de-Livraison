# Prédiction du Temps de Livraison Client (Delivery Time Prediction)

##  Objectif du Projet
Ce projet vise à développer un **modèle de régression Machine Learning** capable de **prédire avec précision le temps total d’une livraison (`Delivery_Time_min`)** pour une entreprise de logistique.

### Objectifs spécifiques :
-  Anticiper les retards et optimiser la planification des livraisons.  
-  Informer les clients en temps réel avec des estimations fiables.  
-  Explorer et analyser les données (EDA) pour comprendre l’influence de facteurs tels que le **trafic**, la **météo**, la **distance** ou encore **l’expérience du livreur**.  
-  Entraîner et comparer deux modèles de régression : **RandomForestRegressor** et **SVR (Support Vector Regressor)**.  
-  Évaluer les performances avec les métriques **MAE (Mean Absolute Error)** et **R² (Coefficient de Détermination)**.

---

##  Gestion du Projet
Pour garantir une approche structurée et professionnelle du développement du modèle prédictif :
-  **Gestion des tâches** (EDA, prétraitement, modélisation, tests) réalisée avec **Jira**.  
-  **Versionnement du code** avec **Git et GitHub**, en utilisant des **branches dédiées** (EDA, tests unitaires, pipeline).  
-  **Architecture modulaire** : séparation claire entre le pipeline, les notebooks d’analyse et les tests.

---

## Structure du Projet
La structure suivante présente l’organisation des fichiers et répertoires du projet :  

```bash
Prediction-du-Churn-Client-desabonnement-/

│── data/                # Données CSV 
│── pipeline.py/         # Script du pipeline de Machine Learning
│── main.ipynb/      # Notebooks d’analyse exploratoire
│── test_pipeline.py/    # Tests unitaires avec Pytest
│   
│── requirements.txt     # Liste des dépendances
│── README.md            # Documentation du projet
```

##  Technologies Utilisées
- **Python**
- **Pandas**, **NumPy** → manipulation et préparation des données  
- **Matplotlib**, **Seaborn** → visualisation et analyse exploratoire  
- **Scikit-learn** →  
  - Prétraitement : `StandardScaler`, `OneHotEncoder`  
  - Sélection de variables : `SelectKBest (f_regression)`  
  - Modélisation : `RandomForestRegressor`, `SVR`  
  - Optimisation : `GridSearchCV`
- **Pytest / Unittest** → tests automatisés de qualité et de performance

---

##  Installation et Exécution

### 1 -> Cloner le dépôt
```bash
git clone https://github.com/manalfarouq/Prediction-du-Temps-de-Livraison.git
cd Prediction-du-Temps-de-Livraison
```

### 2 -> Installer les dépendances
```bash
pip install -r requirements.txt
```

### 3 ->  Exécuter le projet

```bash
jupyter notebook main.ipynb
```


Pipeline Machine Learning complet :

```bash
python pipeline.py
```


Tests automatisés (validation des données et MAE) :

```bash
pytest -v test_pipeline.py
```

##  Résultats et Choix du Modèle Final

| Modèle                     | MAE (min) | R²    |
|-----------------------------|-----------|-------|
| **SVR (Support Vector Regressor)** | **5.835** | **0.821** |
| RandomForestRegressor       | 6.980     | 0.768 |

###  Conclusion
Le **SVR** a été sélectionné comme modèle final car :

- **MAE plus faible** (≈ 5.8 min → erreur moyenne ≈ 5 min 50 s)  
- **Meilleure capacité explicative** (R² = 82.1 %)  
- **Excellente stabilité** après optimisation par **GridSearchCV**

> Malgré une interprétation légèrement plus complexe que celle du Random Forest, le SVR offre une **précision supérieure**, le rendant idéal pour fournir des estimations fiables aux clients.

## Bonus : Intégration Continue avec GitHub Actions

Pour garantir que le code reste **toujours fonctionnel**, on peut configurer **GitHub Actions** pour exécuter automatiquement les tests à chaque push sur la branche `main`.

### 🔹 Configuration

Crée un fichier dans ton dépôt :  

```bash
.github/workflows/python-tests.yml
```
Avec le contenu suivant :
```yaml
name: Run Unit Tests

# Déclenche le workflow à chaque push sur main
on:
  push:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest  # Environnement Ubuntu récent
    steps:
      - name: Checkout du code
        uses: actions/checkout@v4

      - name: Installer Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.14'

      - name: Installer les dépendances
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest joblib scikit-learn pandas

      - name: Exécuter les tests
        run: pytest -v
```

## 🔹 Comment ça fonctionne

1. **Déclenchement automatique**  
   À chaque push sur la branche `main`, GitHub lance le workflow automatiquement.

2. **Installation de Python et des dépendances**  
   Un environnement Python propre est créé et toutes les librairies nécessaires sont installées.

3. **Exécution des tests**  
   Les tests unitaires définis dans `test_pipeline.py` sont exécutés automatiquement.

4. **Alertes en cas d’erreur**  
   Si un test échoue, GitHub affiche une alerte rouge dans l’onglet **Actions**, ce qui permet de corriger rapidement les problèmes.

## 🔹 Avantages

- **Stabilité du code**  
  Garantit que le code reste stable et fonctionnel à chaque modification.

- **Détection rapide des erreurs**  
  Permet de détecter rapidement les erreurs avant qu’elles n’affectent le projet.

- **Collaboration facilitée**  
  Utile pour collaborer avec plusieurs développeurs sur le même dépôt.
