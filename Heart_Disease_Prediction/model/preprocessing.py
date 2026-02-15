# model/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


def load_data(path):
    return pd.read_csv(path)


def preprocess_data(df):

    df = df.copy()

    # -----------------------------
    # TARGET PROCESSING
    # -----------------------------
    df = df.dropna(subset=["num"]).copy()
    df["num"] = df["num"].apply(lambda x: 0 if x == 0 else 1).astype(int)

    y = df["num"]
    X = df.drop("num", axis=1)

    # Drop unnecessary columns
    cols_to_drop = ['id', 'dataset']
    X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])

    # -----------------------------
    # IDENTIFY COLUMN TYPES
    # -----------------------------
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(include="object").columns.tolist()

    # -----------------------------
    # IMPUTATION
    # -----------------------------
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    if len(num_cols) > 0:
        X[num_cols] = num_imputer.fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    # -----------------------------
    # LABEL ENCODING
    # -----------------------------
    encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le

    feature_columns = X.columns.tolist()

    return X, y, encoders, num_imputer, cat_imputer, feature_columns, num_cols
