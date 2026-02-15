# model/train_models.py

import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from preprocessing import load_data, preprocess_data

# -----------------------------
# Load Data
# -----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "dataset", "heart_disease_uci.csv")

df = load_data(data_path)

X, y, encoders, num_imputer, cat_imputer, feature_columns, num_cols = preprocess_data(df)

# -----------------------------
# Save preprocessing artifacts
# -----------------------------
model_dir = os.path.join(project_root, "trained_models")
os.makedirs(model_dir, exist_ok=True)

joblib.dump(encoders, os.path.join(model_dir, "encoders.pkl"))
joblib.dump(num_imputer, os.path.join(model_dir, "num_imputer.pkl"))
joblib.dump(feature_columns, os.path.join(model_dir, "feature_columns.pkl"))
joblib.dump(num_cols, os.path.join(model_dir, "num_columns.pkl"))

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

joblib.dump((X_test, y_test), os.path.join(model_dir, "test_data.pkl"))

# -----------------------------
# Scaling
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

# -----------------------------
# Models
# -----------------------------
models = {
    "logistic": LogisticRegression(max_iter=1000),
    "decision_tree": DecisionTreeClassifier(),
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "random_forest": RandomForestClassifier(),
    "xgboost": XGBClassifier(eval_metric="logloss")
}

for name, model in models.items():

    if name in ["logistic", "knn"]:
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)

    joblib.dump(model, os.path.join(model_dir, f"{name}_model.pkl"))

print("✅ Training complete.")
