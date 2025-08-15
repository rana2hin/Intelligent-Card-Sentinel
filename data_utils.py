import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler

# Target column names used across layers
TARGET_FRAUD = "Is_Fraud"
TARGET_BILLING_ERROR = "Is_Billing_Error"
WINDOW_SIZES_HOURS = [1, 6, 24, 168]


def load_datasets(
    csv_path: str = "simulated_credit_card_transactions.csv",
    oversample: bool = True,
) -> dict:
    """Load the raw CSV and return processed train/val/test splits.

    The function performs the following steps:
        * parse timestamps and sort chronologically
        * define numerical and categorical feature lists (including velocity windows)
        * create a preprocessing pipeline with imputation, scaling and one hot encoding
        * split the data into train/validation/test keeping chronological order
        * fit the preprocessor on the training set and transform all splits
        * compute class weights for fraud and billing error targets to address
          class imbalance
        * optionally oversample the training split so downstream models see a
          more balanced class distribution (improves recall for rare events)

    Parameters
    ----------
    csv_path: str
        Path to the transaction CSV file.

    Returns
    -------
    dict
        Dictionary containing DataFrames, numpy arrays, targets and the fitted
        preprocessor.
    """
    df = pd.read_csv(Path(csv_path))
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.sort_values("Timestamp").reset_index(drop=True)

    categorical_features = [
        "Merchant_Category_Code",
        "Point_of_Sale_Entry_Mode",
        "Transaction_Currency_Code",
        "Transaction_Country_Code",
        "Country_Code_CH",
        "Persona_Type",
        "Merchant_Risk_Level",
        "AVS_Response_Code",
        "CVV_Match_Result",
        "Billing_Error_Type",
        "Transaction_DayOfWeek",
    ]

    numerical_features = [
        "Transaction_Amount_Local_Currency",
        "Is_Card_Present",
        "Is_Cross_Border_Transaction",
        "Credit_Limit",
        "Reported_Fraud_History_Count",
        "Billing_Dispute_History_Count",
        "Historical_Fraud_Rate_Global",
        "Historical_Billing_Dispute_Rate_Global",
        "CH_Avg_Amount",
        "CH_Median_Amount",
        "CH_StdDev_Amount",
        "CH_Transaction_Amount_ZScore",
        "CH_Frequency_MCC_Usage",
        "CH_Count_Transactions_per_Day",
        "Time_Since_CH_Last_Transaction_Overall_Min",
        "Time_Since_CH_Last_Transaction_at_Same_Merchant_Min",
        "Transaction_Hour",
    ]

    # Velocity window features
    for window_hr in WINDOW_SIZES_HOURS:
        numerical_features += [
            f"CH_Count_Transactions_Last_{window_hr}H",
            f"CH_Sum_Amount_Transactions_Last_{window_hr}H",
            f"CH_Count_Unique_Merchants_Last_{window_hr}H",
        ]

    numerical_features = [c for c in numerical_features if c in df.columns]
    categorical_features = [c for c in categorical_features if c in df.columns]

    # Ensure categoricals are strings and fill missing
    for col in categorical_features:
        df[col] = df[col].astype(str).fillna("Missing")

    features = numerical_features + categorical_features

    # chronological split train/test then validation from train
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    train_df, val_df = train_test_split(train_df, test_size=0.2, shuffle=False)

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", num_pipe, numerical_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
    )
    preprocessor.fit(train_df[features])

    X_train = preprocessor.transform(train_df[features])
    X_val = preprocessor.transform(val_df[features])
    X_test = preprocessor.transform(test_df[features])

    y_train_fraud = train_df[TARGET_FRAUD].values
    y_val_fraud = val_df[TARGET_FRAUD].values
    y_test_fraud = test_df[TARGET_FRAUD].values

    y_train_billing = train_df[TARGET_BILLING_ERROR].values
    y_val_billing = val_df[TARGET_BILLING_ERROR].values
    y_test_billing = test_df[TARGET_BILLING_ERROR].values

    # Optionally oversample the minority class on the training set.  We keep
    # separate resampled matrices for the fraud and billing error targets so
    # that each downstream model can train on a balanced dataset tailored to
    # its objective.
    if oversample:
        ros = RandomOverSampler(random_state=42)
        X_train_fraud, y_train_fraud = ros.fit_resample(X_train, y_train_fraud)
        X_train_billing, y_train_billing = ros.fit_resample(X_train, y_train_billing)
    else:
        X_train_fraud, y_train_fraud = X_train, y_train_fraud
        X_train_billing, y_train_billing = X_train, y_train_billing

    # Compute class weights for downstream models to mitigate class imbalance
    fraud_classes = np.unique(y_train_fraud)
    fraud_weights = compute_class_weight(
        class_weight="balanced", classes=fraud_classes, y=y_train_fraud
    )
    billing_classes = np.unique(y_train_billing)
    billing_weights = compute_class_weight(
        class_weight="balanced",
        classes=billing_classes,
        y=y_train_billing,
    )

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train_fraud": X_train_fraud,
        "X_train_billing": X_train_billing,
        "X_val": X_val,
        "X_test": X_test,
        "y_train_fraud": y_train_fraud,
        "y_val_fraud": y_val_fraud,
        "y_test_fraud": y_test_fraud,
        "y_train_billing": y_train_billing,
        "y_val_billing": y_val_billing,
        "y_test_billing": y_test_billing,
        "fraud_class_weights": dict(zip(fraud_classes, fraud_weights)),
        "billing_class_weights": dict(zip(billing_classes, billing_weights)),
        "preprocessor": preprocessor,
        "features": features,
    }
