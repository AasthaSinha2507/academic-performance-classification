import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("StudentsPerformance.csv")

# Create pass/fail column
df["average_score"] = df[["math score", "reading score", "writing score"]].mean(axis=1)
df["pass_fail"] = df["average_score"].apply(lambda x: 1 if x >= 50 else 0)

# Encode categorical columns
gender_map = {"male": 1, "female": 0}
lunch_map = {"standard": 1, "free/reduced": 0}
testprep_map = {"completed": 1, "none": 0}
race_map = {"group A": 0, "group B": 1, "group C": 2, "group D": 3, "group E": 4}
edu_map = {
    "associate's degree": 0,
    "bachelor's degree": 1,
    "master's degree": 2,
    "high school": 3,
    "some college": 4,
    "some high school": 5
}

df["gender"] = df["gender"].map(gender_map)
df["lunch"] = df["lunch"].map(lunch_map)
df["test preparation course"] = df["test preparation course"].map(testprep_map)
df["race/ethnicity"] = df["race/ethnicity"].map(race_map)
df["parental level of education"] = df["parental level of education"].map(edu_map)

# Features and target
X = df[["gender", "race/ethnicity", "parental level of education", "lunch",
        "test preparation course", "math score", "reading score", "writing score"]]
y = df["pass_fail"]

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")