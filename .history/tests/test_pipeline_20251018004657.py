# ==============================================================
# 🚀 Pipeline complet : Optimisation + Tests Automatisés (MAE)
# ==============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# --------------------------------------------------------------
# 1️⃣ Chargement ou génération de données d’exemple
# --------------------------------------------------------------
# Exemple : données fictives
X = pd.DataFrame(np.random.rand(200, 5), columns=[f'feature_{i}' for i in range(5)])
y = pd.Series(np.random.rand(200) * 100)  # variable cible

# --------------------------------------------------------------
# 2️⃣ Vérification du format et des dimensions
# --------------------------------------------------------------
assert isinstance(X, pd.DataFrame), "❌ X doit être un DataFrame"
assert isinstance(y, (pd.Series, np.ndarray)), "❌ y doit être une Série ou un ndarray"
assert X.shape[0] == y.shape[0], "❌ X et y doivent avoir le même nombre de lignes"
assert X.shape[1] > 0, "❌ X doit contenir au moins une colonne"
print("✅ Tests de format et de dimensions : OK")

# --------------------------------------------------------------
# 3️⃣ Séparation en train/test
# --------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------------------
# 4️⃣ Définition des pipelines
# --------------------------------------------------------------
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('Regression', RandomForestRegressor(random_state=42))
])

pipeline_svr = Pipeline([
    ('scaler', StandardScaler()),
    ('SVR', SVR())
])

# --------------------------------------------------------------
# 5️⃣ Définition des grilles d’hyperparamètres
# --------------------------------------------------------------
param_rf = {
    'Regression__n_estimators': [100, 200],
    'Regression__max_depth': [None, 10, 20]
}

param_svr = {
    'SVR__C': [0.1, 1, 10],
    'SVR__kernel': ['linear', 'rbf'],
    'SVR__gamma': ['scale', 'auto']
}

# --------------------------------------------------------------
# 6️⃣ Optimisation par GridSearchCV
# --------------------------------------------------------------
grid_rf = GridSearchCV(pipeline_rf, param_rf, cv=5, scoring='neg_mean_absolute_error')
grid_svr = GridSearchCV(pipeline_svr, param_svr, cv=5, scoring='neg_mean_absolute_error')

grid_rf.fit(X_train, y_train)
grid_svr.fit(X_train, y_train)

# --------------------------------------------------------------
# 7️⃣ Récupération des meilleurs modèles et MAE de validation croisée
# --------------------------------------------------------------
best_model_rf = grid_rf.best_estimator_
best_model_svr = grid_svr.best_estimator_

best_mae_rf_cv = -grid_rf.best_score_
best_mae_svr_cv = -grid_svr.best_score_

print("\n" + "="*60)
print("🔍 RÉSULTATS DE L'OPTIMISATION PAR VALIDATION CROISÉE")
print("="*60)
print(f"RandomForestRegressor :")
print(f"  ➤ Meilleurs hyperparamètres : {grid_rf.best_params_}")
print(f"  ➤ Meilleur MAE (CV) : {best_mae_rf_cv:.2f}")
print("-" * 30)
print(f"SVR :")
print(f"  ➤ Meilleurs hyperparamètres : {grid_svr.best_params_}")
print(f"  ➤ Meilleur MAE (CV) : {best_mae_svr_cv:.2f}")
print("="*60)

# --------------------------------------------------------------
# 8️⃣ Évaluation sur le jeu de test
# --------------------------------------------------------------
y_pred_rf = best_model_rf.predict(X_test)
y_pred_svr = best_model_svr.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
mae_svr = mean_absolute_error(y_test, y_pred_svr)

print("\n📊 ÉVALUATION SUR LE JEU DE TEST")
print(f"RandomForestRegressor - MAE : {mae_rf:.2f}")
print(f"SVR - MAE : {mae_svr:.2f}")

# --------------------------------------------------------------
# 9️⃣ Test automatique de performance (seuil MAE max)
# --------------------------------------------------------------
MAE_SEUIL = 10.0  # par exemple : 10 minutes maximum

assert mae_rf <= MAE_SEUIL, f"❌ RandomForestRegressor MAE ({mae_rf:.2f}) dépasse le seuil ({MAE_SEUIL})"
assert mae_svr <= MAE_SEUIL, f"❌ SVR MAE ({mae_svr:.2f}) dépasse le seuil ({MAE_SEUIL})"

print("\n✅ Tests de performance : MAE sous le seuil acceptable")
print("🚀 Tous les tests automatisés ont réussi avec succès !")
