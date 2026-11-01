import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Wine Quality ML Classifier Suite",
    page_icon="🍷",
    layout="wide"
)

st.title("🍷 Wine Quality Classification & Model Evaluation Portal")
st.markdown(
    "Evaluate five pre-trained machine learning classifiers on the **Wine Quality** test dataset "
    "without retraining. Upload `test_data.csv` in the sidebar to inspect metrics, "
    "confusion matrices, and classification reports."
)

@st.cache_resource
def load_saved_artifacts():
    metadata = joblib.load("model/metadata.joblib")
    model_paths = {
        "Logistic Regression": "model/logistic_regression.joblib",
        "Decision Tree": "model/decision_tree.joblib",
        "KNN": "model/knn.joblib",
        "Naive Bayes": "model/naive_bayes.joblib",
        "Support Vector Machine": "model/support_vector_machine.joblib"
    }
    models = {name: joblib.load(path) for name, path in model_paths.items()}
    return metadata, models

try:
    metadata, models = load_saved_artifacts()
    target_col = metadata["target_col"]
    feature_cols = metadata["feature_cols"]
except Exception as e:
    st.error(f"Model artifacts not found in model/ folder. Run `python train_models.py` first. Details: {e}")
    st.stop()

with st.sidebar:
    st.header("⚙️ Evaluation Controls")
    selected_model = st.selectbox(
        "1. Choose Pre-Trained Classifier",
        options=list(models.keys()),
        index=4
    )
    uploaded_csv = st.file_uploader(
        "2. Upload Test CSV (test_data.csv)",
        type=["csv"]
    )
    st.divider()
    st.caption("BITS Pilani WILP — M.Tech (AIML) ML Assignment 2")

def preprocess_uploaded_df(df_in):
    df_clean = df_in.copy()
    if "Id" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Id"])
    if "total_acidity" not in df_clean.columns and {"fixed acidity", "volatile acidity"}.issubset(df_clean.columns):
        df_clean["total_acidity"] = df_clean["fixed acidity"] + df_clean["volatile acidity"]
    if target_col in df_clean.columns and df_clean[target_col].max() > 1:
        df_clean[target_col] = (df_clean[target_col] >= 6).astype(int)
    return df_clean

def calculate_six_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC Score": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC Score": matthews_corrcoef(y_true, y_pred)
    }

eval_df = None
if uploaded_csv is not None:
    eval_df = preprocess_uploaded_df(pd.read_csv(uploaded_csv))
    st.success(f"Loaded `{uploaded_csv.name}` ({len(eval_df)} test instances).")
elif os.path.exists("test_data.csv"):
    st.info("👈 Upload `test_data.csv` from the sidebar, or inspect the default repository test set below.")
    if st.checkbox("Load default repository test_data.csv", value=True):
        eval_df = preprocess_uploaded_df(pd.read_csv("test_data.csv"))

if eval_df is not None:
    missing = [c for c in feature_cols + [target_col] if c not in eval_df.columns]
    if missing:
        st.error(f"Uploaded file is missing required columns: {missing}")
    else:
        X_test_eval = eval_df[feature_cols]
        y_test_eval = eval_df[target_col]

        clf_pipeline = models[selected_model]
        y_pred = clf_pipeline.predict(X_test_eval)
        y_prob = clf_pipeline.predict_proba(X_test_eval)[:, 1]

        scores = calculate_six_metrics(y_test_eval, y_pred, y_prob)

        st.subheader(f"📊 Performance Metrics: {selected_model}")
        metric_cols = st.columns(6)
        for idx, (label, val) in enumerate(scores.items()):
            metric_cols[idx].metric(label=label, value=f"{val:.4f}")

        st.divider()
        tab_cm, tab_all, tab_data = st.tabs([
            "🧩 Confusion Matrix & Classification Report",
            "🏆 All 5 Models Benchmark on Test Data",
            "🔍 Uploaded Test Data Preview"
        ])

        with tab_cm:
            left_col, right_col = st.columns([1, 1.3])
            with left_col:
                st.markdown("**Confusion Matrix**")
                cm = confusion_matrix(y_test_eval, y_pred)
                fig, ax = plt.subplots(figsize=(4.8, 3.8))
                sns.heatmap(
                    cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False,
                    xticklabels=["Standard (<6)", "Good (>=6)"],
                    yticklabels=["Standard (<6)", "Good (>=6)"],
                    ax=ax
                )
                ax.set_xlabel("Predicted Wine Class")
                ax.set_ylabel("Actual Wine Class")
                ax.set_title(selected_model)
                st.pyplot(fig)

            with right_col:
                st.markdown("**Classification Report**")
                rep = classification_report(
                    y_test_eval, y_pred,
                    target_names=["Standard (0)", "Good (1)"],
                    output_dict=True, zero_division=0
                )
                st.dataframe(pd.DataFrame(rep).transpose().style.format("{:.4f}"), use_container_width=True)

        with tab_all:
            st.markdown("**Comparison of All 5 Saved Models on Current Test Data**")
            summary_rows = []
            for m_name, m_pipe in models.items():
                preds = m_pipe.predict(X_test_eval)
                probs = m_pipe.predict_proba(X_test_eval)[:, 1]
                m_metrics = calculate_six_metrics(y_test_eval, preds, probs)
                row = {"ML Model Name": m_name}
                row.update({k: round(v, 4) for k, v in m_metrics.items()})
                summary_rows.append(row)
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

        with tab_data:
            st.dataframe(eval_df.head(25), use_container_width=True)