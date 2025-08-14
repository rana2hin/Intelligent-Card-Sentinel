"""Decision and action engine combining layer outputs with a meta learner."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import TARGET_BILLING_ERROR, TARGET_FRAUD, load_datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_meta_dataset(df):
    features = [
        "Layer1_Reconstruction_Error",
        "Layer2_Fraud_Probability",
        "Layer3_Billing_Error_Probability",
        "Transaction_Amount_Local_Currency",
    ]
    for f in features:
        if f not in df.columns:
            df[f] = 0.0
    X = df[features].astype(float).values
    y = ((df[TARGET_FRAUD] == 1) | (df[TARGET_BILLING_ERROR] == 1)).astype(int).values
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, shuffle=False
    )
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float()
    )
    return train_ds, test_ds, scaler


class MetaNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x).squeeze(-1)


def train_model(train_ds: TensorDataset, test_ds: TensorDataset):
    model = MetaNet(train_ds.tensors[0].shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    for epoch in range(50):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    model.eval()
    xb, yb = test_ds.tensors
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    with torch.no_grad():
        probs = torch.sigmoid(model(xb)).cpu().numpy()
    y_true = yb.cpu().numpy()
    preds = (probs > 0.5).astype(int)
    report = classification_report(y_true, preds, output_dict=False)
    try:
        auc_score = roc_auc_score(y_true, probs)
    except ValueError:
        auc_score = float("nan")
    return model, probs, preds, y_true, report, auc_score


def main() -> None:
    data = load_datasets()
    df = data["test_df"].copy()
    train_ds, test_ds, _ = build_meta_dataset(df)
    model, probs, preds, y_true, report, auc_score = train_model(train_ds, test_ds)
    print("Layer 4 meta-learner performance:\n", report)
    print(f"ROC AUC: {auc_score:.4f}")

    meta_df = df.iloc[len(train_ds) :].copy().reset_index(drop=True)
    meta_df["Layer4_Final_Risk_Probability"] = probs
    meta_df["Layer4_Final_Risk_Prediction"] = preds
    meta_df["Suggested_Action"] = "Approve"
    meta_df.loc[probs > 0.4, "Suggested_Action"] = "Flag_For_Review"
    meta_df.loc[probs > 0.7, "Suggested_Action"] = "Decline_Or_StepUp"
    print(meta_df[[
        "Layer1_Reconstruction_Error",
        "Layer2_Fraud_Probability",
        "Layer3_Billing_Error_Probability",
        TARGET_FRAUD,
        TARGET_BILLING_ERROR,
        "Layer4_Final_Risk_Probability",
        "Suggested_Action",
    ]].head(20))


if __name__ == "__main__":
    main()
