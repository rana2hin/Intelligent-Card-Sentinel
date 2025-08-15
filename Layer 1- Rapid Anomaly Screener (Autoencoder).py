# ==============================================================================
# --- Layer 1: Rapid Anomaly Screener (Autoencoder) - IMPROVED VERSION ---
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from data_utils import TARGET_BILLING_ERROR, TARGET_FRAUD, load_datasets

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# Import Optuna for hyperparameter tuning
try:
    import optuna
except ImportError:
    print("Optuna not found. Please install it: pip install optuna")
    # If running in a notebook without internet, this will fail.
    # In that case, you'd have to manually tune parameters.
    exit()

print("\n--- Layer 1: Rapid Anomaly Screener (Autoencoder) - IMPROVED WORKFLOW ---")

# Load data
data = load_datasets(oversample=False)
train_df = data["train_df"]
val_df = data["val_df"]
test_df = data["test_df"]

# Further split training data for validation specific to Layer 1
train_df_l1, val_df_l1 = train_test_split(
    train_df,
    test_size=0.2,
    shuffle=False,
)

print(f"Original Train shape: {train_df.shape}")
print(f"New L1 Train shape: {train_df_l1.shape}")
print(f"New L1 Validation shape: {val_df_l1.shape}")
print(f"Final Test shape: {test_df.shape}")


# --- 2. Feature Selection & Preprocessing Setup ---
# We'll use the same feature set as before.
layer1_numerical_features = [
    'Transaction_Amount_Local_Currency', 'Is_Card_Present',
    'CH_Avg_Amount', 'CH_Median_Amount', 'CH_StdDev_Amount', 'CH_Transaction_Amount_ZScore',
    'CH_Frequency_MCC_Usage', 'CH_Count_Transactions_per_Day', 'Transaction_Hour'
]
layer1_categorical_features = [
    'Merchant_Category_Code', 'Point_of_Sale_Entry_Mode', 'Transaction_Currency_Code',
    'Transaction_Country_Code', 'Persona_Type', 'Transaction_DayOfWeek'
]
layer1_numerical_features = [f for f in layer1_numerical_features if f in train_df.columns]
layer1_categorical_features = [f for f in layer1_categorical_features if f in train_df.columns]

# Create preprocessor with MinMaxScaler, which is better for Autoencoders
numerical_transformer_minmax = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor_l1_minmax = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_minmax, layer1_numerical_features),
        ('cat', categorical_transformer, layer1_categorical_features)
    ],
    remainder='drop'
)

# Prepare the data splits
# IMPORTANT: Train the preprocessor ONLY on the training data.
normal_train_df_l1 = train_df_l1[(train_df_l1[TARGET_FRAUD] == 0) & (train_df_l1[TARGET_BILLING_ERROR] == 0)]
X_train_normal_scaled = preprocessor_l1_minmax.fit_transform(normal_train_df_l1)

# Transform validation and test sets
X_val_scaled = preprocessor_l1_minmax.transform(val_df_l1)
y_val_fraud = val_df_l1[TARGET_FRAUD] # We need the "ground truth" for optimization

X_test_scaled = preprocessor_l1_minmax.transform(test_df)
y_test_fraud = test_df[TARGET_FRAUD]
y_test_billing_error = test_df[TARGET_BILLING_ERROR]

input_dim = X_train_normal_scaled.shape[1]
print(f"Input dimension for Autoencoder: {input_dim}")


# --- 3. Hyperparameter Optimization with Optuna ---

def objective(trial):
    """
    This function will be called by Optuna to train and evaluate a model.
    Optuna will try to maximize the return value of this function.
    """
    # Suggest hyperparameters to Optuna
    n_layers = trial.suggest_int('n_layers', 1, 3) # Number of encoder/decoder hidden layers
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Build the model
    model_layers = [Input(shape=(input_dim,))]
    
    # Encoder layers
    last_layer_neurons = input_dim
    for i in range(n_layers):
        neurons = trial.suggest_int(f'n_units_encoder_l{i}', 32, 256, log=True)
        # Ensure layers get smaller
        neurons = min(neurons, last_layer_neurons)
        model_layers.append(Dense(neurons, activation='relu'))
        model_layers.append(Dropout(dropout_rate))
        last_layer_neurons = neurons

    # Decoder layers (mirrors the encoder)
    for i in range(n_layers - 1, -1, -1):
        neurons = model_layers[i * 2 + 1].units # Get neuron count from corresponding encoder layer
        model_layers.append(Dense(neurons, activation='relu'))
        model_layers.append(Dropout(dropout_rate))

    model_layers.append(Dense(input_dim, activation='sigmoid'))
    
    model = Sequential(model_layers)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
    
    # Train the model
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    model.fit(
        X_train_normal_scaled, X_train_normal_scaled,
        epochs=50, # Shorter epochs for faster tuning
        batch_size=256,
        shuffle=True,
        validation_split=0.1, # Inner validation split for training stability
        callbacks=[early_stopping],
        verbose=0 # Suppress output during tuning
    )
    
    # Evaluate on our dedicated validation set
    X_val_pred = model.predict(X_val_scaled, verbose=0)
    val_mse = np.mean(np.power(X_val_scaled - X_val_pred, 2), axis=1)
    
    # Calculate PR-AUC, our optimization metric
    precision, recall, _ = precision_recall_curve(y_val_fraud, val_mse)
    pr_auc_score = auc(recall, precision)
    
    return pr_auc_score

# Create and run the Optuna study
print("\nStarting Hyperparameter Optimization with Optuna...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20) # Run 20 different trials to find the best params

print("\nOptimization finished.")
print("Best trial PR-AUC:", study.best_value)
print("Best hyperparameters found:", study.best_params)

# --- 4. Build, Train, and Evaluate the Best Model ---

# Get the best parameters from the study
best_params = study.best_params
best_n_layers = best_params['n_layers']
best_learning_rate = best_params['learning_rate']
best_dropout_rate = best_params['dropout_rate']

# Build the final model architecture with the best params
final_model_layers = [Input(shape=(input_dim,))]
last_layer_neurons = input_dim
for i in range(best_n_layers):
    neurons = best_params[f'n_units_encoder_l{i}']
    final_model_layers.append(Dense(neurons, activation='relu'))
    final_model_layers.append(Dropout(best_dropout_rate))
    last_layer_neurons = neurons

for i in range(best_n_layers - 1, -1, -1):
    neurons = final_model_layers[i * 2 + 1].units
    final_model_layers.append(Dense(neurons, activation='relu'))
    final_model_layers.append(Dropout(best_dropout_rate))

final_model_layers.append(Dense(input_dim, activation='sigmoid'))

final_autoencoder = Sequential(final_model_layers)
final_autoencoder.compile(optimizer=Adam(learning_rate=best_learning_rate), loss='mse')
final_autoencoder.summary()

# Train the final model on ALL normal training data (no validation split needed here)
print("\nTraining final best model on all normal training data...")
final_autoencoder.fit(
    X_train_normal_scaled, X_train_normal_scaled,
    epochs=100, # Train for longer now that we have the best params
    batch_size=256,
    shuffle=True,
    callbacks=[EarlyStopping(monitor='loss', patience=10)], # Monitor training loss directly
    verbose=1
)

# --- 5. Find Optimal Threshold on Validation Set ---
X_val_pred_final = final_autoencoder.predict(X_val_scaled)
val_mse_final = np.mean(np.power(X_val_scaled - X_val_pred_final, 2), axis=1)

precision, recall, thresholds = precision_recall_curve(y_val_fraud, val_mse_final)
f1_scores = np.nan_to_num(2 * (precision * recall) / (precision + recall))

best_threshold_idx = np.argmax(f1_scores)
OPTIMIZED_THRESHOLD = thresholds[best_threshold_idx]
print(f"\nOptimal Anomaly Threshold (found on validation set): {OPTIMIZED_THRESHOLD:.6f}")


# --- 6. Final Evaluation on the Unseen Test Set ---
print("\n--- Final Performance Evaluation on Test Set ---")
X_test_pred = final_autoencoder.predict(X_test_scaled)
test_mse = np.mean(np.power(X_test_scaled - X_test_pred, 2), axis=1)

# Add results to the main test_df for analysis and use by Layer 4
test_df['Layer1_Reconstruction_Error'] = test_mse
test_df['Layer1_Is_Anomaly_Flag'] = (test_df['Layer1_Reconstruction_Error'] > OPTIMIZED_THRESHOLD).astype(int)

# Plot final distributions
plt.figure(figsize=(10, 6))
sns.histplot(data=test_df, x='Layer1_Reconstruction_Error', hue=TARGET_FRAUD, bins=50, kde=True)
plt.title('Final Distribution of Reconstruction Errors on Test Set')
plt.axvline(OPTIMIZED_THRESHOLD, color='r', linestyle='--', label=f'Optimal Threshold ({OPTIMIZED_THRESHOLD:.4f})')
plt.legend()
plt.show()

# Boxplots
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=test_df[TARGET_FRAUD], y=test_df['Layer1_Reconstruction_Error'])
plt.title('Reconstruction Error vs Actual Fraud')

plt.subplot(1, 2, 2)
sns.boxplot(x=test_df[TARGET_BILLING_ERROR], y=test_df['Layer1_Reconstruction_Error'])
plt.title('Reconstruction Error vs Actual Billing Error')
plt.tight_layout()
plt.show()

# Final flagging report on test set
print("\nFinal Layer 1 Anomaly Flagging Report (on Test Set):")
print("Comparison with Actual Fraud:")
print(pd.crosstab(test_df['Layer1_Is_Anomaly_Flag'], y_test_fraud))
print("\nComparison with Actual Billing Errors:")
print(pd.crosstab(test_df['Layer1_Is_Anomaly_Flag'], y_test_billing_error))

# Standard classification metrics treating fraud as the positive class
y_pred = test_df['Layer1_Is_Anomaly_Flag'].values
print("\nClassification metrics vs Fraud labels:")
print(classification_report(y_test_fraud, y_pred, target_names=['Legit', 'Fraud']))

cm = confusion_matrix(y_test_fraud, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Layer 1 Confusion Matrix')
plt.tight_layout()
plt.show()
