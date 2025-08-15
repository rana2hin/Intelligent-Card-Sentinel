"""Layer 3 - Billing Anomaly Detector
-------------------------------------
Detects billing errors using a Balanced Random Forest classifier.  The
training set is oversampled for billing error labels to maximise recall.
The model evaluates both binary billing-error detection and provides
standard performance plots.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV

from data_utils import load_datasets, TARGET_BILLING_ERROR


def main() -> None:
    data = load_datasets()
    X_train = data["X_train_billing"]
    y_train = data["y_train_billing"]
    X_test = data["X_test"]
    y_test = data["y_test_billing"]

    param_dist = {
        "n_estimators": [200, 400, 600],
        "max_depth": [8, 12, 16],
        "min_samples_split": [2, 5, 10],
    }

    clf = BalancedRandomForestClassifier(random_state=42, n_jobs=-1)
    search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=10,
        scoring="recall",
        cv=3,
        random_state=42,
        verbose=0,
        n_jobs=-1,
    )

    print("Fitting Balanced Random Forest (optimising recall)...")
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)

    model = search.best_estimator_
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    print("\nLayer 3 performance on test set:")
    print(classification_report(y_test, y_pred, target_names=["No Error", "Billing Error"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["No Error", "Billing Error"], yticklabels=["No Error", "Billing Error"])
    plt.title("Layer 3 Confusion Matrix")
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
    plt.title("Layer 3 ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Layer 3 Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
