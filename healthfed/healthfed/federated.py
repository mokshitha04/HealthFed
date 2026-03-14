import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ============================================================
# Load Datasets
# ============================================================
h1 = pd.read_csv("data/hospital1.csv")
h2 = pd.read_csv("data/hospital2.csv")
h3 = pd.read_csv("data/hospital3.csv")
full = pd.read_csv("data/diabetes.csv")

# ============================================================
# Train Local Model Function
# ============================================================
def train_local(data):

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    return model.coef_, model.intercept_

# ============================================================
# Centralized Model (Baseline)
# ============================================================
X_full = full.drop("Outcome", axis=1)
y_full = full["Outcome"]

X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

central_model = LogisticRegression(max_iter=1000)
central_model.fit(X_train_full, y_train_full)

central_preds = central_model.predict(X_test_full)
centralized_acc = accuracy_score(y_test_full, central_preds)

print("Centralized Accuracy:", centralized_acc)

# ============================================================
# Federated Learning + Privacy Tradeoff
# ============================================================
rounds = 5
noise_levels = [0, 0.01, 0.05, 0.1]
privacy_accuracies = []

for noise_level in noise_levels:

    print("\n=== Testing Noise Level:", noise_level, "===")

    for r in range(rounds):

        w1, b1 = train_local(h1)
        w2, b2 = train_local(h2)
        w3, b3 = train_local(h3)

        w1 += np.random.normal(0, noise_level, w1.shape)
        w2 += np.random.normal(0, noise_level, w2.shape)
        w3 += np.random.normal(0, noise_level, w3.shape)

        global_weights = (w1 + w2 + w3) / 3
        global_bias = (b1 + b2 + b3) / 3

    temp_model = LogisticRegression(max_iter=1000)
    temp_model.fit(X_train_full, y_train_full)

    temp_model.coef_ = global_weights
    temp_model.intercept_ = global_bias

    preds = temp_model.predict(X_test_full)
    acc = accuracy_score(y_test_full, preds)

    privacy_accuracies.append(acc)

    print("Accuracy:", acc)

# ============================================================
# Privacy vs Accuracy Graph
# ============================================================
plt.figure()
plt.plot(noise_levels, privacy_accuracies, marker='o')
plt.title("Privacy vs Accuracy Tradeoff")
plt.xlabel("Noise Level (Privacy Strength)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# ============================================================
# Select Best Noise Level
# ============================================================
best_index = np.argmax(privacy_accuracies)
best_noise = noise_levels[best_index]

print("\nBest Noise Level:", best_noise)
print("Best Federated Accuracy:", max(privacy_accuracies))

# ============================================================
# Convergence Graph
# ============================================================
round_accuracies = []

for r in range(rounds):

    w1, b1 = train_local(h1)
    w2, b2 = train_local(h2)
    w3, b3 = train_local(h3)

    w1 += np.random.normal(0, best_noise, w1.shape)
    w2 += np.random.normal(0, best_noise, w2.shape)
    w3 += np.random.normal(0, best_noise, w3.shape)

    global_weights = (w1 + w2 + w3) / 3
    global_bias = (b1 + b2 + b3) / 3

    temp_model = LogisticRegression(max_iter=1000)
    temp_model.fit(X_train_full, y_train_full)

    temp_model.coef_ = global_weights
    temp_model.intercept_ = global_bias

    preds = temp_model.predict(X_test_full)
    acc = accuracy_score(y_test_full, preds)

    round_accuracies.append(acc)

plt.figure()
plt.plot(range(1, rounds+1), round_accuracies, marker='o')
plt.title("Federated Learning Convergence")
plt.xlabel("Communication Round")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# ============================================================
# Hospital Performance Comparison
# ============================================================
def evaluate_local(data):

    X = data.drop("Outcome", axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    return accuracy_score(y_test, preds)

h1_acc = evaluate_local(h1)
h2_acc = evaluate_local(h2)
h3_acc = evaluate_local(h3)

plt.figure(figsize=(8,5))
plt.bar(
    ["Hospital1","Hospital2","Hospital3","Centralized","Federated"],
    [h1_acc, h2_acc, h3_acc, centralized_acc, max(privacy_accuracies)]
)

plt.title("Performance Comparison")
plt.ylabel("Accuracy")
plt.show()

# ============================================================
# Final Global Model
# ============================================================
global_model = LogisticRegression(max_iter=1000)
global_model.fit(X_train_full, y_train_full)

global_model.coef_ = global_weights
global_model.intercept_ = global_bias

joblib.dump(global_model, "diabetes_model.pkl")

print("\nFederated model saved as diabetes_model.pkl")

# ============================================================
# Sample Prediction
# ============================================================
sample_patient = X_full.iloc[0:1]

prediction = global_model.predict(sample_patient)

if prediction[0] == 1:
    print("\nPrediction: High Diabetes Risk")
else:
    print("\nPrediction: Low Diabetes Risk")

# ============================================================
# SHAP Explainability
# ============================================================
explainer = shap.LinearExplainer(global_model, X_full)

shap_values = explainer(X_full)

print("\nShowing SHAP explanation for first patient...")

shap.plots.waterfall(shap_values[0], show=True)

plt.show()