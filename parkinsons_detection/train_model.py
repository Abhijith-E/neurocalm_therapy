# train_dummy_model.py
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

os.makedirs("models", exist_ok=True)
MODEL_PATH = os.path.join("models", "gait_parkinsons_model.pkl")

# Synthetic dummy dataset: 100 samples, 10 features
np.random.seed(42)
normal = np.random.rand(50, 10) * 0.4                  # lower values
parkinson = np.random.rand(50, 10) * 0.4 + 0.6         # higher values

X = np.vstack([normal, parkinson])
y = np.array([0] * 50 + [1] * 50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Dummy model accuracy (on synthetic data): {accuracy_score(y_test, y_pred)*100:.2f}%")

joblib.dump(clf, MODEL_PATH)
print(f"Saved dummy gait model to: {MODEL_PATH}")
