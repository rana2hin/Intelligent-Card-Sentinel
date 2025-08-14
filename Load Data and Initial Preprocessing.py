"""Data loading and initial preprocessing utilities.

This script leverages :mod:`data_utils` to build train, validation and
 test splits that downstream layers can reuse.  The intent is to keep data
 preparation in a single place so models can focus purely on learning.
"""
from __future__ import annotations

from data_utils import TARGET_BILLING_ERROR, TARGET_FRAUD, load_datasets


def main() -> None:
    data = load_datasets()
    train_df = data["train_df"]
    val_df = data["val_df"]
    test_df = data["test_df"]

    print("Dataset loaded successfully.")
    print(f"Train shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(
        "Train time range:",
        f"{train_df['Timestamp'].min()} -> {train_df['Timestamp'].max()}",
    )
    print(
        "Validation time range:",
        f"{val_df['Timestamp'].min()} -> {val_df['Timestamp'].max()}",
    )
    print(
        "Test time range:",
        f"{test_df['Timestamp'].min()} -> {test_df['Timestamp'].max()}",
    )

    print("Targets available:")
    print(train_df[[TARGET_FRAUD, TARGET_BILLING_ERROR]].head())


if __name__ == "__main__":
    main()
