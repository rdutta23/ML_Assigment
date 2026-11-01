import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)

# Customize with the last 4 digits of your BITS ID for unique model outputs
RANDOM_SEED = 4219
os.makedirs("model", exist_ok=True)

# 1. Load attached WineQT.csv dataset
df = pd.read_csv("training_data/full_dataset.csv")

# Drop arbitrary row index 'Id' and create 12th chemical feature 'total_acidity'
if "Id" in df.columns:
    df = df.drop(columns=["Id"])

df["total_acidity"] = df["fixed acidity"] + df["volatile acidity"]

# Convert quality (3-8) into binary target: 1 if quality >= 6 (Good), 0 if < 6 (Standard)
TARGET_COL = "quality"
df[TARGET_COL] = (df[TARGET_COL] >= 6).astype(int)

feature_cols = [c for c in df.columns if c != TARGET_COL]
X = df[feature_cols]
y = df[TARGET_COL]

print(f"Dataset Verified: {X.shape[0]} instances, {X.shape[1]} input features")
assert X.shape[0] >= 500, "Minimum 500 instances required!"
assert X.shape[1] >= 12, "Minimum 12 features required!"

# 2. Split dataset and export test_data.csv for Streamlit upload
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)

test_df = X_test.copy()
test_df[TARGET_COL] = y_test
test_df.to_csv("test_data.csv", index=False)
print(f"Exported test_data.csv ({len(test_df)} rows).")

metadata = {
    "target_col": TARGET_COL,
    "feature_cols": feature_cols,
    "classes": [0, 1]
}
joblib.dump(metadata, "model/metadata.joblib")

# 3. Train and evaluate all 5 required classification models
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_SEED),
    "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance"),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=1.5, probability=True, random_state=RANDOM_SEED)
}

results = []
for name, clf in classifiers.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", clf)
    ])
    pipe.fit(X_train, y_train)

    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(pipe, os.path.join("model", filename))

    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "ML Model Name": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "AUC": round(roc_auc_score(y_test, y_prob), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "MCC": round(matthews_corrcoef(y_test, y_pred), 4)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("model/benchmark_results.csv", index=False)

print("\n=== BITS VIRTUAL LAB EXECUTION: MODEL BENCHMARK ===")
print(results_df.to_markdown(index=False))

