import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns

# For Autoencoder (Layer 1)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# For Classification (Layers 2, 3)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, precision_recall_curve, auc, confusion_matrix

# Load the dataset
try:
    df = pd.read_csv("/kaggle/input/simulated-credit-card-transactions/simulated_credit_card_transactions.csv")
except FileNotFoundError:
    print("Error: simulated_credit_card_transactions.csv not found. Make sure you've generated it.")
    exit()

print("Dataset loaded successfully.")
print(f"Shape of dataset: {df.shape}")
# df.head()

# --- Basic Data Cleaning and Type Conversion ---
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Identify categorical and numerical features (initial pass)
# We'll refine this per layer
categorical_features_initial = [
    'Merchant_Category_Code', 'Point_of_Sale_Entry_Mode', 'Transaction_Currency_Code',
    'Transaction_Country_Code', 'Country_Code_CH', # Cardholder's country
    'Persona_Type', 'Merchant_Risk_Level', 'AVS_Response_Code', 'CVV_Match_Result',
    'Billing_Error_Type', 'Transaction_DayOfWeek' # Hour is numerical but can be cyclical
]
# Ensure all potential categoricals are string type for one-hot encoding
for col in categorical_features_initial:
    if col in df.columns:
        df[col] = df[col].astype(str).fillna('Missing') # Fill NaN before one-hot

numerical_features_initial = [
    'Transaction_Amount_Local_Currency', 'Is_Card_Present', 'Is_Cross_Border_Transaction',
    'Credit_Limit', 'Reported_Fraud_History_Count', 'Billing_Dispute_History_Count',
    'Historical_Fraud_Rate_Global', 'Historical_Billing_Dispute_Rate_Global',
    'CH_Avg_Amount', 'CH_Median_Amount', 'CH_StdDev_Amount', 'CH_Transaction_Amount_ZScore',
    'CH_Frequency_MCC_Usage', 'CH_Count_Transactions_per_Day',
    'Time_Since_CH_Last_Transaction_Overall_Min', 'Time_Since_CH_Last_Transaction_at_Same_Merchant_Min',
    'Transaction_Hour'
]
# Add windowed features
window_sizes_hours = [1, 6, 24, 168]
for window_hr in window_sizes_hours:
    numerical_features_initial.append(f'CH_Count_Transactions_Last_{window_hr}H')
    numerical_features_initial.append(f'CH_Sum_Amount_Transactions_Last_{window_hr}H')
    numerical_features_initial.append(f'CH_Count_Unique_Merchants_Last_{window_hr}H')

# Keep only existing columns
numerical_features_initial = [col for col in numerical_features_initial if col in df.columns]
categorical_features_initial = [col for col in categorical_features_initial if col in df.columns]


# --- Time-based Train-Test Split ---
# Sort by timestamp to ensure chronological split
df = df.sort_values('Timestamp').reset_index(drop=True)
split_point = int(len(df) * 0.8) # 80% for training, 20% for testing

train_df = df.iloc[:split_point].copy()
test_df = df.iloc[split_point:].copy()

print(f"Train set shape: {train_df.shape}")
print(f"Test set shape: {test_df.shape}")
print(f"Train time range: {train_df['Timestamp'].min()} to {train_df['Timestamp'].max()}")
print(f"Test time range: {test_df['Timestamp'].min()} to {test_df['Timestamp'].max()}")

# --- Define Target Variables ---
TARGET_FRAUD = 'Is_Fraud'
TARGET_BILLING_ERROR = 'Is_Billing_Error'

# --- Preprocessing Pipelines (will be adapted per layer) ---
# Numerical transformer
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Median is robust to outliers
    ('scaler', StandardScaler()) # Standardize features
])

# Categorical transformer
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse_output=False for easier use with Keras
])

print("Initial data loading and preprocessing setup complete.")
