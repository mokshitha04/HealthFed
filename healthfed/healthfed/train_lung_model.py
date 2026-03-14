import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = pd.read_csv("data/survey lung cancer.csv")

print("Columns in dataset:", data.columns)

target_column = data.columns[-1]   # assume last column is target

X = data.drop(target_column, axis=1)
y = data[target_column]

# Convert categorical data
X = pd.get_dummies(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=2000)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, preds))

joblib.dump(model, "lung_model.pkl")
print("lung_model.pkl saved")