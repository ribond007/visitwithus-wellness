# scripts/train_pipeline.py

import os
import pickle
from pathlib import Path

import pandas as pd
import numpy as np

from huggingface_hub import hf_hub_download

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

import mlflow
import mlflow.sklearn

# Paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_EXPERIMENT = "visitwithus-wellness-experiment"

# -----------------------------
# 1. Load data from HF dataset
# -----------------------------
def load_data():
    """
    Download tourism.csv from Hugging Face Dataset Hub
    and load it into a DataFrame.
    """
    path = hf_hub_download(
        repo_id="RishiBond/visitwithus-wellness-data",
        filename="tourism.csv",
        repo_type="dataset"
    )
    df = pd.read_csv(path)
    return df

# -----------------------------
# 2. Basic cleaning + split
# -----------------------------
def preprocess_and_split(df, y_col="ProdTaken", test_size=0.2, random_state=42):
    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna(how="all", axis=1)

    # Fill numeric / categorical NAs
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")

    # Target as int
    df[y_col] = df[y_col].astype(int)

    X = df.drop(columns=[y_col])
    y = df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    # Save splits locally (for later upload to HF by upload_to_hf.py)
    train_df = X_train.copy()
    train_df[y_col] = y_train
    test_df = X_test.copy()
    test_df[y_col] = y_test

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    return X_train, X_test, y_train, y_test

# -----------------------------
# 3. Preprocessor
# -----------------------------
def build_preprocessor(X):
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Remove ID-like columns if present
    for c in ["CustomerID"]:
        if c in num_features:
            num_features.remove(c)
        if c in cat_features:
            cat_features.remove(c)

    num_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    # For sklearn >= 1.2 use sparse_output=False
    cat_transformer = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_features),
            ("cat", cat_transformer, cat_features)
        ],
        remainder="drop",
        sparse_threshold=0
    )

    return preprocessor

# -----------------------------
# 4. Train & evaluate models
# -----------------------------
def train_and_evaluate(X_train, X_test, y_train, y_test):
    preprocessor = build_preprocessor(X_train)

    models = {
        "logistic": LogisticRegression(max_iter=200, class_weight="balanced"),
        "rf": RandomForestClassifier(random_state=42, n_jobs=-1),
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
        "ada": AdaBoostClassifier(random_state=42)
    }

    params = {
        "logistic": {"classifier__C": [0.1, 1, 5]},
        "rf": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [5, 10, None]
        },
        "xgb": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 6]
        },
        "gb": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [3, 6]
        },
        "ada": {"classifier__n_estimators": [50, 100]}
    }

    best_models = {}
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for name, estimator in models.items():
        pipe = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE()),
            ("classifier", estimator)
        ])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        grid = GridSearchCV(
            pipe,
            params[name],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
            error_score="raise"
        )

        with mlflow.start_run(run_name=f"train_{name}"):
            grid.fit(X_train, y_train)

            best = grid.best_estimator_
            y_pred_proba = best.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)

            mlflow.log_param("model_name", name)
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("roc_auc", float(auc))

            clf_r = classification_report(
                y_test,
                best.predict(X_test),
                output_dict=True
            )
            mlflow.log_metric("precision_1", clf_r.get("1", {}).get("precision", 0))
            mlflow.log_metric("recall_1", clf_r.get("1", {}).get("recall", 0))

            print(f"{name} AUC: {auc:.4f}")

            # Save each candidate model as artifact
            model_path = MODEL_DIR / f"{name}_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(best, f)
            mlflow.log_artifact(str(model_path), artifact_path=f"models/{name}")

            best_models[name] = (best, auc)

    # Pick best by AUC
    best_name, best_info = max(best_models.items(), key=lambda x: x[1][1])
    best_model = best_info[0]

    final_path = MODEL_DIR / "best_model.pkl"
    with open(final_path, "wb") as f:
        pickle.dump(best_model, f)

    print(f"Best model: {best_name} with AUC {best_info[1]:.4f}")
    return final_path

# -----------------------------
# 5. Main
# -----------------------------
if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    model_path = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("Training complete. Model saved to:", model_path)
