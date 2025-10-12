# ==========================================================
# Star Classifier — Neutron vs Quark Stars (Readable Labels)
# ==========================================================
# Run locally:
#     streamlit run star_auto_train_predict_manual_loading_select_model_labels_final.py
# ==========================================================

import os, warnings, time, re
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# ----------------------------------------------------------
st.set_page_config(page_title="🌟 Neutron vs Quark Star Classifier", layout="wide")

st.title("🌟 Neutron vs Quark Star Classifier — Train, Predict & Manual Input")

st.markdown("""
Upload your data or enter values manually:

- **4 columns (mass, radius, k2, star_type)** → trains all models and saves them  
- **3 columns (mass, radius, k2)** → predicts with best saved model  
- Or enter values manually to get instant prediction from a model of your choice  
""")

uploaded_file = st.file_uploader("📂 Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
run_btn = st.button("🚀 Process Data")

# ----------------------------------------------------------
# Manual prediction sidebar
# ----------------------------------------------------------
st.sidebar.header("🔢 Manual Input for Prediction")
manual_mass = st.sidebar.number_input("Mass", value=1.0, format="%.4f")
manual_radius = st.sidebar.number_input("Radius", value=1.0, format="%.4f")
manual_k2 = st.sidebar.text_input("K2 (spectral type)", value="K2")

available_models = ["Best Model", "Decision_Tree", "Random_Forest", "KNN", "Neural_Network"]
manual_model_choice = st.sidebar.selectbox("Choose Model for Manual Prediction", available_models)
manual_btn = st.sidebar.button("🔮 Predict Manual Sample")

# ----------------------------------------------------------
# Label mapping
# ----------------------------------------------------------
LABEL_MAP = {0: "Neutron Star", 1: "Quark Star"}

# ----------------------------------------------------------
# Utility functions
# ----------------------------------------------------------
def normalize_column_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]", "", name)
    if "mass" in name:
        return "mass"
    elif "radiu" in name or "radius" in name:
        return "radius"
    elif "k2" in name:
        return "k2"
    elif "type" in name:
        return "star_type"
    elif "Type" in name:
        return "star_type"
    elif "star" in name:
        return "star_type"
    return name

def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    return df

def load_dataframe(file):
    if file is None:
        return None
    try:
        if file.name.endswith(".csv"):
            return pd.read_csv(file)
        else:
            return pd.read_excel(file)
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
        return None

def preprocess_training_data(df):
    df = df.copy()
    encoders = {}
    for col in df.columns:
        if df[col].dtype == "object":
            enc = LabelEncoder()
            df[col] = enc.fit_transform(df[col].astype(str))
            encoders[col] = enc

    X = df.drop(columns=["star_type"]).values
    y = df["star_type"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler, encoders

def plot_confusion(y_true, y_pred, name):
    labels = [LABEL_MAP[i] for i in sorted(LABEL_MAP.keys())]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax)
    plt.title(f"Confusion Matrix — {name}")
    st.pyplot(fig)

# ----------------------------------------------------------
# MAIN LOGIC
# ----------------------------------------------------------
if run_btn and uploaded_file is not None:
    df = load_dataframe(uploaded_file)
    if df is None:
        st.stop()

    st.subheader("📄 Data Preview (first 5 rows)")
    st.dataframe(df.head())

    # Normalize columns
    df = normalize_dataframe_columns(df)
    st.write(f"✅ Normalized columns: {list(df.columns)}")

    # TRAINING MODE -------------------------------------------------------
    if set(["mass", "radius", "k2", "star_type"]).issubset(df.columns):
        st.subheader("🧠 Training Mode Activated")

        progress = st.progress(0)
        status_text = st.empty()

        X_train, X_test, y_train, y_test, X_train_s, X_test_s, scaler, encoders = preprocess_training_data(df)
        models = {
            "Decision_Tree": DecisionTreeClassifier(random_state=42),
            "Random_Forest": RandomForestClassifier(random_state=42),
            "KNN": KNeighborsClassifier(),
            "Neural_Network": MLPClassifier(max_iter=500, random_state=42)
        }

        results = []
        os.makedirs("models", exist_ok=True)
        total_models = len(models)

        for i, (name, model) in enumerate(models.items(), start=1):
            status_text.text(f"🔹 Training model {i}/{total_models}: {name}...")
            Xtr = X_train_s if name in ["KNN", "Neural_Network"] else X_train
            Xte = X_test_s if name in ["KNN", "Neural_Network"] else X_test

            model.fit(Xtr, y_train)
            y_pred = model.predict(Xte)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='macro')
            rec = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            results.append((name, acc, prec, rec, f1))

            joblib.dump(model, f"models/{name}.joblib")
            plot_confusion(y_test, y_pred, name)

            progress.progress(i / total_models)
            time.sleep(0.3)

        progress.empty()
        status_text.text("✅ All models trained successfully!")

        res_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1"])
        best_idx = res_df["Accuracy"].idxmax()

        def highlight_best(s):
            return ['background-color: lightgreen' if i == best_idx else '' for i in range(len(s))]

        st.dataframe(
            res_df.style.apply(highlight_best, axis=0).format({
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}"
            })
        )

        joblib.dump(scaler, "models/scaler.joblib")
        best_name = res_df.loc[best_idx, "Model"]
        best_model = joblib.load(f"models/{best_name}.joblib")
        joblib.dump(best_model, "models/best_model.joblib")

        st.success(f"🏆 Best Model: {best_name}")

        plt.figure(figsize=(6, 4))
        plt.bar(res_df["Model"], res_df["Accuracy"], color="royalblue")
        plt.title("Model Accuracy Comparison")
        st.pyplot(plt)

    # PREDICTION MODE ----------------------------------------------------
    elif set(["mass", "radius", "k2"]).issubset(df.columns):
        st.subheader("🔮 Prediction Mode Activated")

        if not os.path.exists("models/best_model.joblib"):
            st.error("❌ No trained model found. Please train models first with labeled data.")
            st.stop()

        X = df.copy()
        for col in X.columns:
            if X[col].dtype == "object":
                enc = LabelEncoder()
                X[col] = enc.fit_transform(X[col].astype(str))

        best_model = joblib.load("models/best_model.joblib")
        scaler = joblib.load("models/scaler.joblib")
        X_scaled = scaler.transform(X)

        probs = best_model.predict_proba(X_scaled)
        preds = np.argmax(probs, axis=1)

        df["Predicted_Star_Type"] = [LABEL_MAP[p] for p in preds]
        df["Prob_Neutron_Star"] = probs[:, 0]
        df["Prob_Quark_Star"] = probs[:, 1]

        st.success("✅ Predictions Ready!")
        st.dataframe(df.head())

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Predictions", csv, "star_predictions.csv", "text/csv")

    else:
        st.warning("⚠️ Dataset must have 3 (mass, radius, k2) or 4 (+star_type) columns.")

# ----------------------------------------------------------
# MANUAL PREDICTION SECTION
# ----------------------------------------------------------
if manual_btn:
    model_path = "models/best_model.joblib" if manual_model_choice == "Best Model" else f"models/{manual_model_choice}.joblib"

    if not os.path.exists(model_path):
        st.error(f"❌ Model '{manual_model_choice}' not found. Please train models first.")
    else:
        model = joblib.load(model_path)
        scaler = joblib.load("models/scaler.joblib")

        sample = pd.DataFrame({
            "mass": [manual_mass],
            "radius": [manual_radius],
            "k2": [manual_k2]
        })

        for col in sample.columns:
            if sample[col].dtype == "object":
                enc = LabelEncoder()
                sample[col] = enc.fit_transform(sample[col].astype(str))

        X_scaled = scaler.transform(sample)
        probs = model.predict_proba(X_scaled)[0]
        pred_class = np.argmax(probs)

        st.success(f"🌟 Predicted Star Type: **{LABEL_MAP[pred_class]}** (using {manual_model_choice})")

        fig, ax = plt.subplots()
        ax.bar(LABEL_MAP.values(), probs, color="skyblue")
        ax.set_ylabel("Probability")
        ax.set_title(f"Prediction Probabilities ({manual_model_choice})")
        st.pyplot(fig)
