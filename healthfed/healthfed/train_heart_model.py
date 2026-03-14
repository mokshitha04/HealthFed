import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# --------------------------------------------------
# Load dataset
# --------------------------------------------------
data = pd.read_csv("data/heart.csv")

print("Columns:", data.columns)

# --------------------------------------------------
# Target column
# --------------------------------------------------
y = data["target"]

# --------------------------------------------------
# Features
# --------------------------------------------------
X = data.drop("target", axis=1)

# --------------------------------------------------
# Handle missing values
# --------------------------------------------------
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# --------------------------------------------------
# Train / Test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# Train model
# --------------------------------------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# --------------------------------------------------
# Evaluate model
# --------------------------------------------------
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)

print("Heart Model Accuracy:", accuracy)

# --------------------------------------------------
# Save model
# --------------------------------------------------
joblib.dump(model, "heart_model.pkl")

print("heart_model.pkl saved successfully")