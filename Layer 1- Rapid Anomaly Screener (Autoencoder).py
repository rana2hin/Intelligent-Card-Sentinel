"""GPU optimised autoencoder for rapid anomaly screening.

The model is trained only on normal transactions (no fraud and no billing errors) and learns to reconstruct them. Transactions with high reconstruction error are flagged as anomalies. Hyperparameters are tuned with Optuna and the implementation uses PyTorch for efficient GPU execution.
"""
from __future__ import annotations

import numpy as np
import optuna
import torch
from sklearn.metrics import auc, precision_recall_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from data_utils import (
    TARGET_BILLING_ERROR,
    TARGET_FRAUD,
    load_datasets,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(trial: optuna.Trial, input_dim: int) -> nn.Module:
    """Construct an autoencoder based on trial suggestions."""
    n_layers = trial.suggest_int("n_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    dims = []
    last_dim = input_dim
    for i in range(n_layers):
        units = trial.suggest_int(f"n_units_l{i}", 32, 256, log=True)
        units = min(units, last_dim)
        dims.append(units)
        last_dim = units

    layers = []
    prev = input_dim
    for h in dims:
        layers.extend([nn.Linear(prev, h), nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    for h in reversed(dims[:-1]):
        layers.extend([nn.Linear(prev, h), nn.ReLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev = h
    layers.append(nn.Linear(prev, input_dim))
    return nn.Sequential(*layers).to(DEVICE)


def objective(trial: optuna.Trial) -> float:
    data = load_datasets()
    X_train = data["X_train"]
    X_val = data["X_val"]
    y_val = data["y_val_fraud"]

    normal_idx = np.where(
        (data["y_train_fraud"] == 0) & (data["y_train_billing"] == 0)
    )[0]
    X_train_norm = X_train[normal_idx]

    train_ds = TensorDataset(torch.from_numpy(X_train_norm).float())
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    model = build_model(trial, X_train.shape[1])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(40):
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_tensor = torch.from_numpy(X_val).float().to(DEVICE)
        recon = model(val_tensor)
        mse = torch.mean((val_tensor - recon) ** 2, dim=1).cpu().numpy()
    precision, recall, _ = precision_recall_curve(y_val, mse)
    return auc(recall, precision)


def train_final_model(data: dict, best_params: dict):
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_val = data["y_val_fraud"]
    y_test_fraud = data["y_test_fraud"]
    y_test_billing = data["y_test_billing"]

    normal_idx = np.where(
        (data["y_train_fraud"] == 0) & (data["y_train_billing"] == 0)
    )[0]
    X_train_norm = X_train[normal_idx]

    model = build_model(optuna.trial.FixedTrial(best_params), X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params["lr"])
    loss_fn = nn.MSELoss()
    train_ds = TensorDataset(torch.from_numpy(X_train_norm).float())
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    model.train()
    for epoch in range(100):
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        val_tensor = torch.from_numpy(X_val).float().to(DEVICE)
        recon_val = model(val_tensor)
        val_mse = torch.mean((val_tensor - recon_val) ** 2, dim=1).cpu().numpy()

    precision, recall, thr = precision_recall_curve(y_val, val_mse)
    denom = precision[:-1] + recall[:-1]
    f1_scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom != 0,
    )
    best_thr = thr[np.argmax(f1_scores)]

    test_tensor = torch.from_numpy(X_test).float().to(DEVICE)
    with torch.no_grad():
        recon_test = model(test_tensor)
        test_mse = (
            torch.mean((test_tensor - recon_test) ** 2, dim=1)
            .detach()
            .cpu()
            .numpy()
        )

    test_df = data["test_df"].copy()
    test_df["Layer1_Reconstruction_Error"] = test_mse
    test_df["Layer1_Is_Anomaly_Flag"] = (test_mse > best_thr).astype(int)

    return test_df, best_thr


def main() -> None:
    print("Running Layer 1 autoencoder on", DEVICE)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best trial:", study.best_value)
    data = load_datasets()
    test_df, threshold = train_final_model(data, study.best_params)
    print(f"Optimal reconstruction threshold: {threshold:.6f}")
    print(test_df[["Layer1_Reconstruction_Error", "Layer1_Is_Anomaly_Flag"]].head())


if __name__ == "__main__":
    main()
