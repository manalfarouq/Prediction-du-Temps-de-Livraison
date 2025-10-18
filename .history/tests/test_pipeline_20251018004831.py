import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error

# ----------------------------
# 1️⃣ Créer des données simples
# ----------------------------
X = pd.DataFrame(np.random.rand(100, 3), columns=['A', 'B', 'C'])
y = pd.Series(np.random.rand(100) * 100)

# ----------------------------
# 2️⃣ Vérification des données
# ----------------------------
assert X.shape[0] == y.shape[0], "❌ X et y n'ont pas la même taille"
print("Données valides")

# ----------------------------
# 3️⃣ Séparation train/test
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 4️⃣ Pipeline + modèle SVR
# ----------------------------
pipeline_svr = Pipeline([
    ('scaler', StandardScaler()),
    ('SVR', SVR())
])

# Grille simple d’hyperparamètres
paramsvr = {
    'SVR__C': [0.1, 1, 10],
    'SVR__kernel': ['linear', 'rbf']
}

# ----------------------------
# 5️⃣ GridSearchCV
# ----------------------------
grid_svr = GridSearchCV(pipeline_svr, paramsvr, cv=5, scoring='neg_mean_absolute_error')
grid_svr.fit(X_train, y_train)

# ----------------------------
# 6️⃣ Meilleurs résultats
# ----------------------------
best_model = grid_svr.best_estimator_
best_mae_cv = -grid_svr.best_score_
print(f"\n✅ Meilleur modèle : {grid_svr.best_params_}")
print(f"✅ Meilleur MAE (CV) : {best_mae_cv:.2f}")

# ----------------------------
# 7️⃣ Évaluation sur test
# ----------------------------
y_pred = best_model.predict(X_test)
mae_test = mean_absolute_error(y_test, y_pred)
print(f"📊 MAE sur le jeu de test : {mae_test:.2f}")

# ----------------------------
# 8️⃣ Test automatique de seuil
# ----------------------------
SEUIL_MAE = 10.0  # par exemple : erreur max 10
assert mae_test <= SEUIL_MAE, f"❌ MAE ({mae_test:.2f}) dépasse le seuil ({SEUIL_MAE})"
print("✅ Test de performance : MAE sous le seuil acceptable")
