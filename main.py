import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Distribution Shift
X_test_shifted = X_test + np.random.normal(0, 0.5, X_test.shape)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    pred = model.predict(X_test_shifted)
    acc = accuracy_score(y_test, pred)

    print(f"\n{name}")
    print("Accuracy:", acc)
    print(classification_report(y_test, pred))

    results[name] = acc * 100

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f"{name} Confusion Matrix")
    plt.savefig(f"graphs/{name}_cm.png")
    plt.close()

# Bar Graph
plt.figure()
plt.bar(results.keys(), results.values())
plt.title("Model Comparison")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=20)
plt.savefig("graphs/model_comparison.png")
plt.show()