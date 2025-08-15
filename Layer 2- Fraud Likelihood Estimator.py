"""Layer 2 - Fraud Likelihood Estimator
--------------------------------------
Trains an XGBoost classifier to predict fraudulent transactions.  The
training split returned by :func:`data_utils.load_datasets` is oversampled
with :class:`imblearn.over_sampling.RandomOverSampler` to maximise recall for
rare fraud events.  Hyper-parameters are tuned with
:class:`sklearn.model_selection.RandomizedSearchCV` optimising recall.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV

from data_utils import load_datasets, TARGET_FRAUD


def main() -> None:
    data = load_datasets()
    X_train = data["X_train_fraud"]
    y_train = data["y_train_fraud"]
    X_test = data["X_test"]
    y_test = data["y_test_fraud"]

    print("Training examples after oversampling:", X_train.shape[0])

    param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 1.0],
    }

    clf = XGBClassifier(
        eval_metric="logloss",
        scale_pos_weight=1,
        n_jobs=-1,
        random_state=42,
    )

    search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=10,
        scoring="recall",
        cv=3,
        verbose=0,
        random_state=42,
        n_jobs=-1,
    )

    print("Fitting XGBoost classifier (optimising recall)...")
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)

    model = search.best_estimator_
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("\nLayer 2 performance on test set:")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
    plt.title("Layer 2 Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Layer 2 ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Layer 2 Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
