import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


def test_pipeline_svr():
    X = pd.DataFrame(np.random.rand(100, 3), columns=['1', '2', 'C'])
    y = pd.Series(np.random.rand(100) * 100)

    assert X.shape[0] == y.shape[0], "X et y n'ont pas la même taille"
    print("Données valides")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline_svr = Pipeline([
        ('scaler', StandardScaler()),
        ('SVR', SVR())
    ])

    paramsvr = {
        'SVR__C': [0.1, 1, 10],
        'SVR__kernel': ['linear', 'rbf']
    }

    grid_svr = GridSearchCV(pipeline_svr, paramsvr, cv=5, scoring='neg_mean_absolute_error')
    grid_svr.fit(X_train, y_train)

    best_model = grid_svr.best_estimator_
    best_mae_cv = -grid_svr.best_score_
    print(f"\nMeilleur modèle : {grid_svr.best_params_}")
    print(f"Meilleur MAE (CV) : {best_mae_cv:.2f}")

    y_pred = best_model.predict(X_test)
    mae_test = mean_absolute_error(y_test, y_pred)
    print(f"MAE = {mae_test:.2f}")

    seuil_mae = 40.0
    assert mae_test <= seuil_mae, f"MAE ({mae_test:.2f}) dépasse le seuil ({seuil_mae})"
    print("Test de performance : MAE sous le seuil acceptable")
