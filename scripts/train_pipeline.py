# scripts/train_pipeline.py
import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

import mlflow
import mlflow.sklearn

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Config / placeholders (replace HF_USERNAME if you want to log)
HF_USERNAME = os.getenv("HF_USERNAME", "your-hf-username")
MLFLOW_EXPERIMENT = "visitwithus-wellness-experiment"

def load_data():
    df = pd.read_csv(DATA_DIR / "tourism.csv")
    return df

def preprocess_and_split(df, y_col="ProdTaken", test_size=0.2, random_state=42):
    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna(how='all', axis=1)
    # fill numeric/cat NAs
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna('Unknown')
    # Ensure target is int
    df[y_col] = df[y_col].astype(int)
    X = df.drop(columns=[y_col])
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def build_preprocessor(X):
    num_features = X.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    # remove potential ID-like columns
    for c in ['CustomerID']:
        if c in num_features:
            num_features.remove(c)
        if c in cat_features:
            cat_features.remove(c)
    num_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    # Use sparse_output for compatibility with scikit-learn >=1.2
    cat_transformer = Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_features),
        ('cat', cat_transformer, cat_features)
    ], remainder='drop', sparse_threshold=0)
    return preprocessor


def train_and_evaluate(X_train, X_test, y_train, y_test):
    preprocessor = build_preprocessor(X_train)
    models = {
        "logistic": LogisticRegression(max_iter=200, class_weight='balanced'),
        "rf": RandomForestClassifier(random_state=42, n_jobs=-1),
        "xgb": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
        "ada": AdaBoostClassifier(random_state=42)
    }

    params = {
        "logistic": {"classifier__C":[0.1,1,5]},
        "rf": {"classifier__n_estimators":[100,200],"classifier__max_depth":[5,10,None]},
        "xgb": {"classifier__n_estimators":[100,200],"classifier__max_depth":[3,6]},
        "gb": {"classifier__n_estimators":[100,200],"classifier__max_depth":[3,6]},
        "ada": {"classifier__n_estimators":[50,100]}
    }

    best_models = {}
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    for name, estimator in models.items():
        pipe = ImbPipeline(steps=[('preprocessor', preprocessor), ('smote', SMOTE()), ('classifier', estimator)])
         cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
         grid = GridSearchCV(pipe, params[name], cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)
         with mlflow.start_run(run_name=f"train_{name}"):
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            y_pred_proba = best.predict_proba(X_test)[:,1]
            auc = roc_auc_score(y_test, y_pred_proba)
            mlflow.log_param("model_name", name)
            mlflow.log_params({k:v for k,v in grid.best_params_.items()})
            mlflow.log_metric("roc_auc", float(auc))
            # log classification report
            clf_r = classification_report(y_test, best.predict(X_test), output_dict=True)
            mlflow.log_metric("precision_1", clf_r.get('1', {}).get('precision', 0))
            mlflow.log_metric("recall_1", clf_r.get('1', {}).get('recall', 0))
            print(f"{name} AUC: {auc:.4f}")
            best_models[name] = (best, auc)
            # save artifact for this model
            fname = MODEL_DIR / f"{name}_model.pkl"
            with open(fname, "wb") as f:
                pickle.dump(best, f)
            mlflow.log_artifact(str(fname), artifact_path=f"models/{name}")
         
     

        

    # choose best by auc
    best_name, best_val = max(best_models.items(), key=lambda x: x[1][1])
    best_model = best_models[best_name][0]
    # persist final model
    final_path = MODEL_DIR / "best_model.pkl"
    with open(final_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model: {best_name} with AUC {best_val:.4f}")
    return final_path

if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_and_split(df)
    model_path = train_and_evaluate(X_train, X_test, y_train, y_test)
    print("Training complete. Model saved to:", model_path)
