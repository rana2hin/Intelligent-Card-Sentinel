"""Billing anomaly detector leveraging GPU-accelerated XGBoost.

The model predicts whether a transaction is a billing error and,
optionally, the type of billing error. Hyperparameters are optimised
with Optuna to maximise the area under the precision–recall curve.

This script is intended to run on environments with GPU support such as
Kaggle notebooks.
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from data_utils import TARGET_BILLING_ERROR, load_datasets


def objective(trial: optuna.Trial) -> float:
    """Optuna objective function for billing error detection."""

    data = load_datasets()
    X_train = data["X_train"]
    y_train = data["y_train_billing"]
    X_val = data["X_val"]
    y_val = data["y_val_billing"]

    # Compute imbalance ratio for scale_pos_weight
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

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
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 1e-3, 10.0, log=True
        ),
        "scale_pos_weight": scale_pos_weight,
    }

    num_boost = trial.suggest_int("n_estimators", 200, 1000)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    preds = model.predict(dval)
    precision, recall, _ = precision_recall_curve(y_val, preds)
    return auc(recall, precision)


def train_final_model(data: dict, best_params: dict):
    """Train the final billing error model and produce test predictions."""

    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train_billing"]
    y_val = data["y_val_billing"]
    y_test = data["y_test_billing"]

    test_df = data["test_df"].copy()

    X_full = np.vstack([X_train, X_val])
    y_full = np.hstack([y_train, y_val])

    neg, pos = np.bincount(y_full)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    dtrain = xgb.DMatrix(X_full, label=y_full)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        **best_params,
        "objective": "binary:logistic",
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "eval_metric": "aucpr",
        "scale_pos_weight": scale_pos_weight,
    }

    num_boost = params.pop("n_estimators", 500)
    model = xgb.train(params, dtrain, num_boost_round=num_boost)

    proba = model.predict(dtest)
    preds = (proba > 0.5).astype(int)

    test_df["Layer3_Billing_Error_Probability"] = proba
    test_df["Layer3_Billing_Error_Prediction"] = preds

    return test_df, y_test, proba, preds


def train_error_type_model(
    data: dict, best_params: dict, test_df: pd.DataFrame
) -> pd.DataFrame:
    """Train a multi-class model to predict the billing error type.

    Only executes if the training data contains more than one distinct
    ``Billing_Error_Type`` value. Predictions are returned for all test
    transactions but are masked for records not predicted as billing
    errors by the binary model.
    """

    from sklearn.preprocessing import LabelEncoder

    train_df = data["train_df"]
    val_df = data["val_df"]
    test_raw = data["test_df"]
    preprocessor = data["preprocessor"]
    features = data["features"]

    train_be = train_df[train_df[TARGET_BILLING_ERROR] == 1].copy()
    val_be = val_df[val_df[TARGET_BILLING_ERROR] == 1].copy()

    if (
        train_be.empty
        or "Billing_Error_Type" not in train_be
        or train_be["Billing_Error_Type"].nunique() < 2
    ):
        test_df["Layer3_Predicted_Error_Type"] = None
        return test_df

    labeler = LabelEncoder()
    y_train = labeler.fit_transform(train_be["Billing_Error_Type"].astype(str))
    y_val = labeler.transform(val_be["Billing_Error_Type"].astype(str))

    X_train = preprocessor.transform(train_be[features])
    X_val = preprocessor.transform(val_be[features])

    num_classes = len(labeler.classes_)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        **best_params,
        "objective": "multi:softprob",
        "num_class": num_classes,
        "tree_method": "gpu_hist",
        "predictor": "gpu_predictor",
        "eval_metric": "mlogloss",
    }

    num_boost = params.pop("n_estimators", 500)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=num_boost,
        evals=[(dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    X_test_all = preprocessor.transform(test_raw[features])
    dtest_all = xgb.DMatrix(X_test_all)
    preds = np.asarray(model.predict(dtest_all)).argmax(axis=1)
    error_types = labeler.inverse_transform(preds)

    test_df["Layer3_Predicted_Error_Type"] = error_types
    mask_non_error = test_df["Layer3_Billing_Error_Prediction"] == 0
    test_df.loc[mask_non_error, "Layer3_Predicted_Error_Type"] = None

    return test_df


def main() -> None:
    """Run hyperparameter search and train final models for Layer 3."""

    print("Optimising Layer 3 XGBoost model")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)

    data = load_datasets()
    test_df, y_test, proba, preds = train_final_model(data, study.best_params)

    print(
        "Classification report:\n",
        classification_report(
            y_test, preds, target_names=["Not Billing Error", "Billing Error"]
        ),
    )
    roc_auc = roc_auc_score(y_test, proba)
    print(f"ROC AUC: {roc_auc:.4f}")
    precision, recall, _ = precision_recall_curve(y_test, proba)
    pr_auc = auc(recall, precision)
    print(f"PR AUC: {pr_auc:.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, preds))

    # Train optional multi-class model for billing error type
    train_error_type_model(data, study.best_params, test_df)


if __name__ == "__main__":
    main()

