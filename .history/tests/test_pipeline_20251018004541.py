import numpy as np
import pandas as pd

# Exemple : X_train, y_train, X_test, y_test déjà définis

# Test 1 : Vérification des dimensions
assert X_train.shape[0] == y_train.shape[0], "❌ Le nombre d'échantillons d'entraînement ne correspond pas à y_train"
assert X_test.shape[0] == y_test.shape[0], "❌ Le nombre d'échantillons de test ne correspond pas à y_test"

# Test 2 : Vérification du type
assert isinstance(X_train, (pd.DataFrame, np.ndarray)), "❌ X_train doit être un DataFrame ou un array NumPy"
assert isinstance(y_train, (pd.Series, np.ndarray, list)), "❌ y_train doit être un vecteur ou une liste"

# Test 3 : Vérification qu’il n’y a pas de valeurs manquantes
assert not pd.isnull(X_train).values.any(), "❌ X_train contient des valeurs manquantes"
assert not pd.isnull(y_train).any(), "❌ y_train contient des valeurs manquantes"

print("✅ Tests sur le format et les dimensions : OK")
