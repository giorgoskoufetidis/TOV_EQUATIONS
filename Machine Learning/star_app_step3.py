# ==========================================================
# Star Classifier — Hybrid App (Pretrained + Trainable)
# ==========================================================
# Run:
#     streamlit run star_classifier_hybrid_app_final.py
# ==========================================================

import os
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# ----------------------------------------------------------
# Page setup
# ----------------------------------------------------------
st.set_page_config(page_title="🌟 Neutron vs Quark Star — Hybrid Classifier", layout="wide")
st.title("🌟 Neutron vs Quark Star Classifier — Pretrained + Train")
st.markdown("""
Use **pretrained models** immediately — or **upload labeled data** to train and update them.

**Columns expected for training:** `mass`, `radius`, `k2`, `star_type`  
**Columns for batch prediction:** `mass`, `radius`, `k2`
""")

# ----------------------------------------------------------
# Paths and constants
# ----------------------------------------------------------
PRETRAINED_PATH = "pretrained_models"
os.makedirs(PRETRAINED_PATH, exist_ok=True)

SCALER_PATH = os.path.join(PRETRAINED_PATH, "scaler.joblib")
K2_ENCODER_PATH = os.path.join(PRETRAINED_PATH, "k2_encoder.joblib")
BEST_MODEL_PATH = os.path.join(PRETRAINED_PATH, "best_model.joblib")

LABEL_MAP = {0: "Neutron Star", 1: "Quark Star"}
MODEL_FILENAMES = ["Decision_Tree", "Random_Forest", "KNN", "Neural_Network"]

# ----------------------------------------------------------
# Utility functions
# ----------------------------------------------------------
def normalize_columns(df):
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def load_dataframe(upload):
    if upload is None:
        return None
    try:
        if upload.name.endswith(".csv"):
            return pd.read_csv(upload)
        else:
            return pd.read_excel(upload)
    except Exception as e:
        st.error(f"❌ Failed to read file: {e}")
        return None

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=[LABEL_MAP[0], LABEL_MAP[1]]).plot(ax=ax)
    plt.title(title)
    st.pyplot(fig)

def metric_table(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

# ----------------------------------------------------------
# ✅ Smart preprocessing for prediction
# ----------------------------------------------------------
def prepare_features_for_model(df, model_name):
    """
    Encodes and scales features dynamically (no external k2 encoder required).
    - Fits temporary LabelEncoder for k2
    - Applies StandardScaler only for KNN and Neural_Network
    """
    from sklearn.preprocessing import LabelEncoder

    df = df.copy()
    # Keep consistent order
    df = df[["mass", "radius", "k2"]]

    # --- Apply scaling conditionally ---
    models_using_scaler = [ "Neural_Network"]

    if model_name in models_using_scaler and os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        X = scaler.transform(df)
    else:
        X = df.values  # raw features for unscaled models (Decision Tree, Random Forest)

    return X

# ----------------------------------------------------------
# Load pretrained assets
# ----------------------------------------------------------
st.sidebar.header("🧠 Pretrained Models")

loaded_models = {}
if os.path.exists(SCALER_PATH):
    st.sidebar.success("✅ Scaler loaded")
else:
    st.sidebar.warning("⚠️ No scaler.joblib found (OK if your models were trained unscaled)")


# Load available models
for name in MODEL_FILENAMES:
    path = os.path.join(PRETRAINED_PATH, f"{name}.joblib")
    if os.path.exists(path):
        loaded_models[name] = joblib.load(path)

if loaded_models:
    st.sidebar.success(f"✅ Loaded models: {', '.join(loaded_models.keys())}")
else:
    st.sidebar.warning("⚠️ No pretrained models found")

best_model = None
if os.path.exists(BEST_MODEL_PATH):
    best_model = joblib.load(BEST_MODEL_PATH)
    st.sidebar.success("🏆 best_model.joblib loaded")

# ----------------------------------------------------------
# Tabs
# ----------------------------------------------------------
tab_predict, tab_train = st.tabs(["🔮 Predict", "🧠 Train / Retrain"])

# ==========================================================
# 🔮 PREDICT TAB
# ==========================================================
with tab_predict:
    st.subheader("Manual Prediction")

    col1, col2, col3 = st.columns(3)
    with col1:
        mass = st.number_input("Mass", value=1.0, format="%.5f")
    with col2:
        radius = st.number_input("Radius", value=1.0, format="%.5f")
    with col3:
        k2_val = st.number_input("K2 (spectral type)", value=1.0, format="%.5f")

    model_options = ["Best_Model"] if best_model else []
    model_options += list(loaded_models.keys())

    model_choice = st.selectbox("Choose Model", model_options)

    if st.button("🔮 Predict Manual Input"):
        if model_choice == "Best_Model" and best_model:
            model = best_model
            model_name = "Best_Model"
        else:
            model = loaded_models.get(model_choice)
            model_name = model_choice

        if model is None:
            st.error("❌ Model not found.")
        else:
            sample = pd.DataFrame({"mass": [mass], "radius": [radius], "k2": [k2_val]})
            X_prepared = prepare_features_for_model(sample, model_name)
            probs = model.predict_proba(X_prepared)[0]
            pred = int(np.argmax(probs))

            st.success(f"🌠 Predicted: **{LABEL_MAP[pred]}** (using {model_name})")

            fig, ax = plt.subplots()
            ax.bar(LABEL_MAP.values(), probs, color="skyblue")
            ax.set_ylabel("Probability")
            ax.set_title(f"Prediction Probabilities — {model_name}")
            st.pyplot(fig)

    # Batch Prediction
    st.markdown("---")
    st.subheader("📂 Batch Prediction")

    uploaded = st.file_uploader("Upload CSV/XLSX (mass, radius, k2)", type=["csv", "xlsx", "xls"])
    batch_model = st.selectbox("Model for Batch Prediction", model_options, key="batch_model")

    if st.button("🚀 Predict Batch File"):
        if uploaded is None:
            st.error("❌ Please upload a file")
        else:
            df = load_dataframe(uploaded)
            if df is None:
                st.stop()
            df = normalize_columns(df)
            required = {"mass", "radius", "k2"}
            if not required.issubset(df.columns):
                st.error("❌ Columns required: mass, radius, k2")
                st.stop()

            if batch_model == "Best_Model" and best_model:
                model = best_model
                model_name = "Best_Model"
            else:
                model = loaded_models.get(batch_model)
                model_name = batch_model

            if model is None:
                st.error("❌ Selected model not available")
                st.stop()

            X_prepared = prepare_features_for_model(df, model_name)
            probs = model.predict_proba(X_prepared)
            preds = np.argmax(probs, axis=1)

            df["Predicted_Type"] = [LABEL_MAP[p] for p in preds]
            df["Prob_Neutron_Star"] = probs[:, 0]
            df["Prob_Quark_Star"] = probs[:, 1]

            st.success(f"✅ Predictions done using {model_name}")
            st.dataframe(df.head())

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", csv, "star_predictions.csv", "text/csv")

# ==========================================================
# 🧠 TRAIN TAB
# ==========================================================
with tab_train:
    st.subheader("Upload Labeled Data for Training / Retraining")
    file = st.file_uploader("Upload CSV/XLSX (mass, radius, k2, star_type)", type=["csv", "xlsx", "xls"])

    if file:
        df = load_dataframe(file)
        if df is not None:
            df = normalize_columns(df)
            st.dataframe(df.head())
            required = {"mass", "radius", "k2", "star_type"}

            if not required.issubset(df.columns):
                st.error("❌ Missing required columns")
            else:
                # Encode target and k2
                y = LabelEncoder().fit_transform(df["star_type"].astype(str))
                k2_enc = LabelEncoder()
                df["k2"] = k2_enc.fit_transform(df["k2"].astype(str))
                joblib.dump(k2_enc, K2_ENCODER_PATH)
                st.info("✅ Saved k2_encoder.joblib")

                X = df[["mass", "radius", "k2"]]
                X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

                scaler_new = StandardScaler()
                X_train_s = scaler_new.fit_transform(X_train)
                X_test_s = scaler_new.transform(X_test)
                joblib.dump(scaler_new, SCALER_PATH)

                st.info("🔄 Training models (GridSearchCV, 5-fold CV)...")

                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                grids = {
                    "Decision_Tree": {
                        "estimator": DecisionTreeClassifier(random_state=0),
                        "grid": {"max_depth": [None, 5, 10, 20]},
                        "use_scaled": False
                    },
                    "Random_Forest": {
                        "estimator": RandomForestClassifier(random_state=0),
                        "grid": {"n_estimators": [100, 200], "max_depth": [None, 10, 20]},
                        "use_scaled": False
                    },
                    "KNN": {
                        "estimator": KNeighborsClassifier(),
                        "grid": {"n_neighbors": [3, 5, 7], "weights": ["uniform", "distance"]},
                        "use_scaled": True
                    },
                    "Neural_Network": {
                        "estimator": MLPClassifier(max_iter=1000, random_state=0),
                        "grid": {"hidden_layer_sizes": [(50,), (100,)], "activation": ["relu", "tanh"]},
                        "use_scaled": True
                    }
                }

                results = []
                progress = st.progress(0)

                best_acc = 0
                best_model_name = None

                for i, (name, spec) in enumerate(grids.items(), start=1):
                    Xtr = X_train_s if spec["use_scaled"] else X_train
                    Xte = X_test_s if spec["use_scaled"] else X_test

                    gs = GridSearchCV(spec["estimator"], spec["grid"], cv=cv, scoring="accuracy", n_jobs=-1)
                    gs.fit(Xtr, y_train)
                    best = gs.best_estimator_
                    y_pred = best.predict(Xte)
                    metrics = metric_table(y_test, y_pred)
                    results.append({"Model": name, **metrics})

                    joblib.dump(best, os.path.join(PRETRAINED_PATH, f"{name}.joblib"))
                    plot_confusion(y_test, y_pred, f"{name} Confusion Matrix")

                    if metrics["Accuracy"] > best_acc:
                        best_acc = metrics["Accuracy"]
                        best_model_name = name
                        joblib.dump(best, BEST_MODEL_PATH)

                    progress.progress(i / len(grids))
                    time.sleep(0.2)

                st.success("✅ Training complete and models saved.")
                res_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
                st.dataframe(res_df.style.format("{:.4f}"))
                if best_model_name:
                    st.success(f"🏆 Best Model: {best_model_name} (Acc={best_acc:.4f})")
