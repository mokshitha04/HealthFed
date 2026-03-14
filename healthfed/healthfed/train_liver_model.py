import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer

# Load dataset
data = pd.read_csv("data/liver.csv")

# Convert categorical values (Male/Female) to numbers
data = pd.get_dummies(data, drop_first=True)

# Target column
y = data["Dataset"]

# Features
X = data.drop("Dataset", axis=1)

# ---- Handle missing values ----
imputer = SimpleImputer(strategy="mean")
X = imputer.fit_transform(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
print("Liver Model Accuracy:", accuracy_score(y_test, preds))

# Save model
joblib.dump(model, "liver_model.pkl")

print("liver_model.pkl saved successfully")