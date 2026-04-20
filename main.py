import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train baseline model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Test baseline
y_pred = model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred)

print("Baseline Accuracy:", baseline_acc)

# Add noise (distribution shift)
X_test_shifted = X_test + np.random.normal(0, 0.5, X_test.shape)

y_pred_shifted = model.predict(X_test_shifted)
shifted_acc = accuracy_score(y_test, y_pred_shifted)

print("Shifted Accuracy:", shifted_acc)

# Random Forest model
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test_shifted)
rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_acc)

# Plot graph
models = ['Baseline', 'Shifted', 'Random Forest']
scores = [baseline_acc*100, shifted_acc*100, rf_acc*100]

plt.bar(models, scores)
plt.title("Model Performance Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")

plt.savefig("result.png")
plt.show()