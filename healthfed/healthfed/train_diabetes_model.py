import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ----------------------------
# Load Dataset
# ----------------------------
data = pd.read_csv("data/diabetes.csv")

# ----------------------------
# Separate Features and Target
# ----------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Create and Train Model
# ----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# ----------------------------
# Predictions
# ----------------------------
predictions = model.predict(X_test)

# ----------------------------
# Evaluation Metrics
# ----------------------------
accuracy = accuracy_score(y_test, predictions)

print("\nModel trained successfully")
print("Centralized Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ----------------------------
# Save Model (for Flask use)
# ----------------------------
joblib.dump(model, "diabetes_model.pkl")
print("\nModel saved as 'diabetes_model.pkl'")
print("File is running ")