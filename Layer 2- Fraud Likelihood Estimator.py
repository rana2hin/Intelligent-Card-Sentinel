"""Fraud likelihood estimator using GPU accelerated XGBoost.

Hyperparameters are optimised with Optuna. The model consumes the
preprocessed features produced by :mod:`data_utils`.
"""
from __future__ import annotations

import numpy as np
import optuna
import xgboost as xgb
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from data_utils import load_datasets


def objective(trial: optuna.Trial) -> float:
    data = load_datasets()
    X_train = data["X_train"]
    y_train = data["y_train_fraud"]
    X_val = data["X_val"]
    y_val = data["y_val_fraud"]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "objective": "binary:logistic",
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "eval_metric": "aucpr",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-8, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 10.0, log=True),
    }
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=[(dval, "val")],
        verbose_eval=False,
        early_stopping_rounds=20,
    )
    preds = model.predict(dval)
    precision, recall, _ = precision_recall_curve(y_val, preds)
    return auc(recall, precision)


def train_final_model(data: dict, best_params: dict):
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train_fraud"]
    y_val = data["y_val_fraud"]
    y_test = data["y_test_fraud"]

    dtrain_full = xgb.DMatrix(
        np.vstack([X_train, X_val]), label=np.hstack([y_train, y_val])
    )
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {
        **best_params,
        "objective": "binary:logistic",
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "eval_metric": "aucpr",
    }
    model = xgb.train(
        params, dtrain_full, num_boost_round=best_params.get("n_estimators", 500)
    )
    proba = model.predict(dtest)
    preds = (proba > 0.5).astype(int)

    test_df = data["test_df"].copy()
    test_df["Layer2_Fraud_Probability"] = proba
    test_df["Layer2_Fraud_Prediction"] = preds

    return test_df, y_test, proba, preds


def main() -> None:
    print("Optimising Layer 2 XGBoost model")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    data = load_datasets()
    test_df, y_test, proba, preds = train_final_model(data, study.best_params)
    print("Classification report:\n", classification_report(y_test, preds))
    roc_auc = roc_auc_score(y_test, proba)
    print(f"ROC AUC: {roc_auc:.4f}")
    precision, recall, _ = precision_recall_curve(y_test, proba)
    pr_auc = auc(recall, precision)
    print(f"PR AUC: {pr_auc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))


if __name__ == "__main__":
    main()
