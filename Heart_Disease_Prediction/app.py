import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# =====================================================
# Page Config
# =====================================================

st.set_page_config(
    page_title="❤️ Heart Disease Prediction Dashboard",
    layout="wide"
)

# =====================================================
# Load Artifacts
# =====================================================

@st.cache_resource
def load_artifacts():

    models = {
        "Logistic Regression": joblib.load("trained_models/logistic_model.pkl"),
        "Decision Tree": joblib.load("trained_models/decision_tree_model.pkl"),
        "KNN": joblib.load("trained_models/knn_model.pkl"),
        "Naive Bayes": joblib.load("trained_models/naive_bayes_model.pkl"),
        "Random Forest": joblib.load("trained_models/random_forest_model.pkl"),
        "XGBoost": joblib.load("trained_models/xgboost_model.pkl"),
    }

    scaler = joblib.load("trained_models/scaler.pkl")
    num_imputer = joblib.load("trained_models/num_imputer.pkl")
    feature_columns = joblib.load("trained_models/feature_columns.pkl")
    num_cols = joblib.load("trained_models/num_columns.pkl") 
    X_test, y_test = joblib.load("trained_models/test_data.pkl")

    return models, scaler, num_imputer, feature_columns,num_cols, X_test, y_test


models, scaler, num_imputer, feature_columns,num_cols, X_test, y_test = load_artifacts()

# =====================================================
# Sidebar
# =====================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Prediction Dashboard", "Model Evaluation"]
)

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    list(models.keys())
)

selected_model = models[selected_model_name]

# =====================================================
# PAGE 1 — Prediction Dashboard
# =====================================================

if page == "Prediction Dashboard":

    st.title("❤️ Heart Disease Prediction Dashboard")

    st.subheader(f"📊 Model Performance: {selected_model_name}")

    # Scale if needed
    if selected_model_name in ["Logistic Regression", "KNN"]:
        X_eval = scaler.transform(X_test)
    else:
        X_eval = X_test

    y_pred = selected_model.predict(X_eval)
    y_prob = selected_model.predict_proba(X_eval)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

    metric_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])

    st.dataframe(metric_df)
    st.bar_chart(metric_df.set_index("Metric"))

    st.markdown("---")

    st.subheader("🧑 Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 20, 100, 50)
        sex = st.selectbox("Sex", ["Female", "Male"])
        cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
        trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)

    with col2:
        chol = st.number_input("Cholesterol", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
        restecg = st.selectbox("Rest ECG", [0, 1, 2])
        thalach = st.number_input("Max Heart Rate", 60, 220, 150)

    with col3:
        exang = st.selectbox("Exercise Induced Angina", [0, 1])
        oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
        slope = st.selectbox("Slope", [0, 1, 2])
        ca = st.selectbox("Major Vessels", [0, 1, 2, 3, 4])
        thal = st.selectbox("Thal", [0, 1, 2, 3])

    sex = 1 if sex == "Male" else 0

    input_dict = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }

    input_df = pd.DataFrame([input_dict])

    input_df = input_df[feature_columns]

    input_df[num_cols] = num_imputer.transform(input_df[num_cols])

    if st.button("Predict"):

        if selected_model_name in ["Logistic Regression", "KNN"]:
            input_processed = scaler.transform(input_df)
        else:
            input_processed = input_df

        prediction = selected_model.predict(input_processed)[0]
        probability = selected_model.predict_proba(input_processed)[0][1]

        if prediction == 1:
            st.error(
                f"⚠️ High Risk of Heart Disease\nProbability: {probability:.2f}"
            )
        else:
            st.success(
                f"✅ Low Risk of Heart Disease\nProbability: {probability:.2f}"
            )


# =====================================================
# PAGE 2 — Model Evaluation
# =====================================================

if page == "Model Evaluation":

    st.title("📊 Evaluate Model on Uploaded Dataset")

    uploaded_file = st.file_uploader(
        "Upload Test CSV",
        type=["csv"]
    )

    if uploaded_file:

        df = pd.read_csv(uploaded_file)

        st.dataframe(df.head())

        if "target" not in df.columns:
            st.error("Dataset must contain 'target' column")
            st.stop()

        y_true = df["target"]
        X_input = df.drop("target", axis=1)

        X_input = X_input[feature_columns]

        X_input[num_cols] = num_imputer.transform(X_input[num_cols])

        if selected_model_name in ["Logistic Regression", "KNN"]:
            X_processed = scaler.transform(X_input)
        else:
            X_processed = X_input

        y_pred = selected_model.predict(X_processed)

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots()

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"]
        )

        st.pyplot(fig)

        st.subheader("Classification Report")

        report = classification_report(
            y_true,
            y_pred,
            output_dict=True
        )

        st.dataframe(pd.DataFrame(report).transpose())
