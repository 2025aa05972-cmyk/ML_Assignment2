
import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

st.set_page_config(page_title="Fraud Detection App", layout="wide")

st.markdown(
    """
    <style>
    .main {background-color: #f4f6f9;}
    h1 {color: #2c3e50;}
    </style>
    """, unsafe_allow_html=True
)

st.title("💳 Credit Card Fraud Detection System")
st.write("Compare 6 Machine Learning Models with Full Evaluation Metrics")

uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

model_option = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    if "Class" not in df.columns:
        st.error("Dataset must contain 'Class' column.")
    else:
        X = df.drop("Class", axis=1)
        y = df["Class"]

        try:
            model = pickle.load(open(f"model/{model_option}.pkl", "rb"))
        except:
            st.error("Model file not found in model folder.")
            st.stop()

        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]

        st.subheader("📊 Evaluation Metrics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", round(accuracy_score(y, y_pred), 4))
        col2.metric("AUC", round(roc_auc_score(y, y_prob), 4))
        col3.metric("Precision", round(precision_score(y, y_pred), 4))

        col4, col5, col6 = st.columns(3)
        col4.metric("Recall", round(recall_score(y, y_pred), 4))
        col5.metric("F1 Score", round(f1_score(y, y_pred), 4))
        col6.metric("MCC", round(matthews_corrcoef(y, y_pred), 4))

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y, y_pred)

        fig = plt.figure()
        plt.imshow(cm)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.colorbar()
        plt.show()

        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y, y_pred))

else:
    st.info("Upload dataset to evaluate models.")
