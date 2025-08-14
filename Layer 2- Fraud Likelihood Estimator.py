print("\n--- Layer 2: Fraud Likelihood Estimator (Gradient Boosting) ---")

# --- Feature Selection for Layer 2 ---
# More comprehensive set of features, including velocity, merchant risk, CNP details
layer2_numerical_features = [
    'Transaction_Amount_Local_Currency', 'Is_Card_Present', 'Is_Cross_Border_Transaction',
    'Credit_Limit', 'Reported_Fraud_History_Count', # CH specific risk
    'Historical_Fraud_Rate_Global', # Merchant specific risk
    'CH_Avg_Amount', 'CH_Median_Amount', 'CH_StdDev_Amount', 'CH_Transaction_Amount_ZScore',
    'CH_Frequency_MCC_Usage', 'CH_Count_Transactions_per_Day',
    'Time_Since_CH_Last_Transaction_Overall_Min',
    'Time_Since_CH_Last_Transaction_at_Same_Merchant_Min',
    'Transaction_Hour'
]
# Add windowed velocity features
for window_hr in window_sizes_hours: # Defined in Part 1
    layer2_numerical_features.append(f'CH_Count_Transactions_Last_{window_hr}H')
    layer2_numerical_features.append(f'CH_Sum_Amount_Transactions_Last_{window_hr}H')
    layer2_numerical_features.append(f'CH_Count_Unique_Merchants_Last_{window_hr}H')

layer2_categorical_features = [
    'Merchant_Category_Code', 'Point_of_Sale_Entry_Mode', 'Transaction_Currency_Code',
    'Transaction_Country_Code', 'Country_Code_CH',
    'Persona_Type', 'Merchant_Risk_Level',
    'AVS_Response_Code', 'CVV_Match_Result', 'Transaction_DayOfWeek'
]

# Include Layer 1's output if available (assuming Layer 1 ran on train_df too)
# For this prototype, we'll train Layer 2 independently first.
# Later, Layer 1's output on the *training data* could be a feature.
# For now, let's assume Layer 1's output is only on test_df.

# Ensure selected features exist
layer2_numerical_features = [f for f in layer2_numerical_features if f in train_df.columns]
layer2_categorical_features = [f for f in layer2_categorical_features if f in train_df.columns]

# --- Create Preprocessor for Layer 2 ---
# Using StandardScaler for numerical features for GBT
preprocessor_l2 = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, layer2_numerical_features), # Using the StandardScaler pipeline
        ('cat', categorical_transformer, layer2_categorical_features)
    ],
    remainder='passthrough' # Keep other columns for now, or 'drop'
)

# --- Prepare Data for Layer 2 ---
X_train_l2 = train_df.drop(columns=[TARGET_FRAUD, TARGET_BILLING_ERROR, 'Billing_Error_Type', 'Timestamp', 'Transaction_ID', 'Cardholder_ID', 'Merchant_ID', 'IP_Address_of_Transaction'])
y_train_l2 = train_df[TARGET_FRAUD]

X_test_l2 = test_df.drop(columns=[TARGET_FRAUD, TARGET_BILLING_ERROR, 'Billing_Error_Type', 'Timestamp', 'Transaction_ID', 'Cardholder_ID', 'Merchant_ID', 'IP_Address_of_Transaction',
                                  'Layer1_Reconstruction_Error', 'Layer1_Is_Anomaly_Flag']) # Remove L1 outputs before L2 prediction

# We need to ensure that columns dropped from X_train_l2 and X_test_l2 are consistent
# with what preprocessor_l2 expects and that the target columns are not in X.
# Let's be more explicit:
features_l2 = layer2_numerical_features + layer2_categorical_features
X_train_l2 = train_df[features_l2].copy()
y_train_l2 = train_df[TARGET_FRAUD].copy()

X_test_l2 = test_df[features_l2].copy()
y_test_l2_fraud_actual = test_df[TARGET_FRAUD].copy() # Actual fraud labels for evaluation


# Apply preprocessing
X_train_l2_processed = preprocessor_l2.fit_transform(X_train_l2)
X_test_l2_processed = preprocessor_l2.transform(X_test_l2)

print(f"Shape of preprocessed training data for L2: {X_train_l2_processed.shape}")
print(f"Shape of preprocessed test data for L2: {X_test_l2_processed.shape}")

# --- Gradient Boosting Model Definition & Training ---
# Handle class imbalance - Gradient Boosting can struggle with highly imbalanced data
# We can use `scale_pos_weight` (for XGBoost) or class_weight (for scikit-learn's GBT if available, or oversample/undersample)
# For scikit-learn's GradientBoostingClassifier, there isn't a direct scale_pos_weight.
# Let's try a simple GBT first, then consider imbalanced-learn or manual weighting if needed.
fraud_ratio_train = y_train_l2.value_counts(normalize=True)
print(f"Fraud ratio in L2 training data: \n{fraud_ratio_train}")

gbt_l2_model = GradientBoostingClassifier(
    n_estimators=150, # Number of trees
    learning_rate=0.1,
    max_depth=5,       # Max depth of individual trees
    subsample=0.8,     # Fraction of samples used for fitting individual base learners
    random_state=42,
    verbose=0
)

print("\nTraining Layer 2 Gradient Boosting model for Fraud Detection...")
gbt_l2_model.fit(X_train_l2_processed, y_train_l2)
print("Layer 2 model training complete.")

# --- Predictions and Evaluation for Layer 2 ---
y_pred_l2_proba = gbt_l2_model.predict_proba(X_test_l2_processed)[:, 1] # Probability of fraud
y_pred_l2_class = gbt_l2_model.predict(X_test_l2_processed)           # Class prediction (0 or 1)

# Add Layer 2 predictions to test_df
test_df['Layer2_Fraud_Probability'] = y_pred_l2_proba
test_df['Layer2_Fraud_Prediction'] = y_pred_l2_class

print("\nLayer 2 Fraud Detection Performance (on Test Set):")
print(classification_report(y_test_l2_fraud_actual, y_pred_l2_class, target_names=['Not Fraud', 'Fraud']))

try:
    roc_auc_l2 = roc_auc_score(y_test_l2_fraud_actual, y_pred_l2_proba)
    print(f"Layer 2 ROC AUC Score: {roc_auc_l2:.4f}")
except ValueError as e:
    print(f"Could not calculate ROC AUC for Layer 2: {e}")
    roc_auc_l2 = None

# Plot ROC Curve
if roc_auc_l2 is not None:
    fpr, tpr, _ = roc_curve(y_test_l2_fraud_actual, y_pred_l2_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Layer 2 GBT (AUC = {roc_auc_l2:.2f})')
    plt.plot([0, 1], [0, 1], 'k--') # Random guessing line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Layer 2 ROC Curve - Fraud Detection')
    plt.legend(loc='lower right')
    plt.show()

# Plot Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test_l2_fraud_actual, y_pred_l2_proba)
pr_auc_l2 = auc(recall, precision)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f'Layer 2 GBT (PR AUC = {pr_auc_l2:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Layer 2 Precision-Recall Curve - Fraud Detection')
plt.legend(loc='lower left')
plt.show()


# Confusion Matrix
cm_l2 = confusion_matrix(y_test_l2_fraud_actual, y_pred_l2_class)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_l2, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Layer 2 Confusion Matrix - Fraud Detection')
plt.show()

# --- Feature Importance (from GBT model) ---
if hasattr(gbt_l2_model, 'feature_importances_'):
    # Get feature names after one-hot encoding
    try:
        # Get feature names from the preprocessor
        feature_names_l2_processed = list(preprocessor_l2.named_transformers_['num'].get_feature_names_out(layer2_numerical_features)) + \
                                     list(preprocessor_l2.named_transformers_['cat'].get_feature_names_out(layer2_categorical_features))
        
        importances = gbt_l2_model.feature_importances_
        feature_importance_df_l2 = pd.DataFrame({'feature': feature_names_l2_processed, 'importance': importances})
        feature_importance_df_l2 = feature_importance_df_l2.sort_values(by='importance', ascending=False).head(20)

        plt.figure(figsize=(10, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance_df_l2)
        plt.title('Layer 2 - Top 20 Feature Importances (Fraud Detection)')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not plot feature importances for Layer 2: {e}")
        # This can happen if preprocessor structure is complex or feature names aren't easily retrieved.
        # For now, we'll just acknowledge it.
