"""Layer 4 - Decision & Action Engine
------------------------------------
Meta learner combining outputs from previous layers.  It expects the test
DataFrame to already contain:
    * ``Layer1_Reconstruction_Error``
    * ``Layer2_Fraud_Probability``
    * ``Layer3_Billing_Error_Probability``
The meta-target is 1 when a transaction is either fraudulent or a billing
error.  Logistic regression is trained on half of the test set and evaluated
on the remainder.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    auc,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import load_datasets, TARGET_FRAUD, TARGET_BILLING_ERROR


def main() -> None:
    data = load_datasets()
    df = data["test_df"].copy()

    required = {
        "Layer1_Reconstruction_Error",
        "Layer2_Fraud_Probability",
        "Layer3_Billing_Error_Probability",
    }
    if not required.issubset(df.columns):
        raise RuntimeError("Run Layers 1-3 before executing Layer 4")

    df["Meta_Target"] = ((df[TARGET_FRAUD] == 1) | (df[TARGET_BILLING_ERROR] == 1)).astype(int)
    X = df[list(required)]
    y = df["Meta_Target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, shuffle=False)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(class_weight="balanced", solver="liblinear", random_state=42)),
    ])

    print("Training meta-learner...")
    pipe.fit(X_train, y_train)

    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = pipe.predict(X_test)

    print("\nLayer 4 performance:")
    print(classification_report(y_test, y_pred, target_names=["Low Risk", "High Risk"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Layer 4 Confusion Matrix")
    plt.tight_layout()
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Layer 4 ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision, label=f"AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Layer 4 Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
