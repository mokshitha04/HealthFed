import pandas as pd
import os

# Load dataset
data = pd.read_csv("data/diabetes.csv")

# Shuffle data
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into 3 parts
split1 = int(len(data) * 0.33)
split2 = int(len(data) * 0.66)

hospital1 = data[:split1]
hospital2 = data[split1:split2]
hospital3 = data[split2:]

# Save files
hospital1.to_csv("data/hospital1.csv", index=False)
hospital2.to_csv("data/hospital2.csv", index=False)
hospital3.to_csv("data/hospital3.csv", index=False)

print("Data successfully split into 3 hospital datasets.")