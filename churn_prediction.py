"""
AI SaaS Customer Churn Prediction
----------------------------------
End-to-end ML pipeline: load data -> engineer features -> train classifiers
-> evaluate -> save the best model.

Usage:
    python churn_prediction.py --data ai_saas_users.csv
"""

import argparse
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["signup_date", "last_login_date", "churn_date"])
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw columns into model-ready features."""
    df = df.copy()

    # Use last_login_date as the reference "today" so this works on any export date
    reference_date = df["last_login_date"].max()

    df["tenure_days"] = (reference_date - df["signup_date"]).dt.days
    df["days_since_last_login"] = (reference_date - df["last_login_date"]).dt.days

    # Engagement / value ratios — these tend to be strong churn signals
    df["api_calls_per_dollar"] = df["api_calls_per_month"] / df["monthly_spend_usd"].replace(0, np.nan)
    df["ai_features_used_ratio"] = df["ai_features_used"] / df["ai_features_used"].max()

    df["api_calls_per_dollar"] = df["api_calls_per_dollar"].fillna(0)

    return df


FEATURE_COLUMNS_NUMERIC = [
    "monthly_spend_usd",
    "ai_features_used",
    "api_calls_per_month",
    "support_tickets",
    "tenure_days",
    "days_since_last_login",
    "api_calls_per_dollar",
    "ai_features_used_ratio",
]

FEATURE_COLUMNS_CATEGORICAL = [
    "plan_type",
    "industry",
    "company_size",
    "country",
]

TARGET_COLUMN = "churned"


def build_pipeline(model) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), FEATURE_COLUMNS_NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLUMNS_CATEGORICAL),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(name: str, y_true, y_pred, y_proba) -> dict:
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
    }
    print(f"\n=== {name} ===")
    for k, v in metrics.items():
        if k != "model":
            print(f"{k:>10}: {v:.3f}")
    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=["Retained", "Churned"], zero_division=0))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(confusion_matrix(y_true, y_pred))
    return metrics


def get_feature_names(pipeline: Pipeline) -> list:
    preprocessor = pipeline.named_steps["preprocess"]
    num_features = FEATURE_COLUMNS_NUMERIC
    cat_features = list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(FEATURE_COLUMNS_CATEGORICAL)
    )
    return num_features + cat_features


def main(data_path: str, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {data_path} ...")
    df = load_data(data_path)
    df = engineer_features(df)

    X = df[FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)} | Churn rate: {y.mean():.1%}")

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
        ),
    }

    results = []
    fitted_pipelines = {}

    for name, model in candidates.items():
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics = evaluate(name, y_test, y_pred, y_proba)
        results.append(metrics)
        fitted_pipelines[name] = pipeline

    # Pick the best model by ROC-AUC (falls back to F1 if AUC is undefined)
    results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    best_name = results_df.iloc[0]["model"]
    best_pipeline = fitted_pipelines[best_name]

    print(f"\nBest model: {best_name}")
    print(results_df.to_string(index=False))

    # Feature importance (Random Forest only, most interpretable here)
    if "Random Forest" in fitted_pipelines:
        rf_pipeline = fitted_pipelines["Random Forest"]
        importances = rf_pipeline.named_steps["model"].feature_importances_
        feature_names = get_feature_names(rf_pipeline)
        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False)
        print("\nTop churn drivers (Random Forest feature importance):")
        print(importance_df.head(10).to_string(index=False))
        importance_df.to_csv(output_path / "feature_importance.csv", index=False)

    # Save predictions on the full dataset (useful for a dashboard / retention list)
    df["churn_probability"] = best_pipeline.predict_proba(X)[:, 1]
    df[["user_id", "churn_probability", TARGET_COLUMN]].sort_values(
        "churn_probability", ascending=False
    ).to_csv(output_path / "churn_risk_scores.csv", index=False)

    # Persist the best model
    model_path = output_path / "churn_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_pipeline, f)
    print(f"\nSaved best model ({best_name}) to {model_path}")
    print(f"Saved churn risk scores to {output_path / 'churn_risk_scores.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a churn prediction model on SaaS user data.")
    parser.add_argument("--data", type=str, default="ai_saas_users.csv", help="Path to the input CSV file.")
    parser.add_argument("--output_dir", type=str, default="output", help="Where to save model + reports.")
    args = parser.parse_args()
    main(args.data, args.output_dir)
