print("\n--- Layer 3: Billing Anomaly Detector (Random Forest) ---")

# --- Feature Selection for Layer 3 ---
# Focus on features relevant to duplicates, subscriptions, merchant billing history
layer3_numerical_features = [
    'Transaction_Amount_Local_Currency', 'Is_Card_Present',
    'Billing_Dispute_History_Count', # CH specific
    'Historical_Billing_Dispute_Rate_Global', # Merchant specific
    'Time_Since_CH_Last_Transaction_Overall_Min',
    'Time_Since_CH_Last_Transaction_at_Same_Merchant_Min', # Key for duplicates
    'Transaction_Hour',
    # Include Layer 1's output if deemed useful (and if run on training data too)
    # For now, keeping it independent like Layer 2
]
# Add windowed velocity features for general context
for window_hr in window_sizes_hours: # Defined in Part 1
    layer3_numerical_features.append(f'CH_Count_Transactions_Last_{window_hr}H')
    layer3_numerical_features.append(f'CH_Sum_Amount_Transactions_Last_{window_hr}H')


layer3_categorical_features = [
    'Merchant_Category_Code', 'Point_of_Sale_Entry_Mode',
    'Transaction_Country_Code', 'Country_Code_CH',
    'Persona_Type', 'Merchant_Risk_Level', # General risk might correlate with billing issues
    'Transaction_DayOfWeek'
    # 'Transaction_Descriptor_Raw_Text' - if we had more advanced NLP features from it
    # 'Is_Recurring_Payment_Flag' - if we had simulated this (good feature for real data)
    # 'Is_First_Recurring_Payment' - if we had simulated this
]

# Ensure selected features exist
layer3_numerical_features = [f for f in layer3_numerical_features if f in train_df.columns]
layer3_categorical_features = [f for f in layer3_categorical_features if f in train_df.columns]

# --- Create Preprocessor for Layer 3 ---
preprocessor_l3 = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, layer3_numerical_features), # Using StandardScaler
        ('cat', categorical_transformer, layer3_categorical_features)
    ],
    remainder='drop' # Drop other columns not used by this layer
)

# --- Prepare Data for Layer 3 ---
# Target 1: Is_Billing_Error (Binary)
features_l3 = layer3_numerical_features + layer3_categorical_features
X_train_l3 = train_df[features_l3].copy()
y_train_l3_be = train_df[TARGET_BILLING_ERROR].copy()

X_test_l3 = test_df[features_l3].copy()
y_test_l3_be_actual = test_df[TARGET_BILLING_ERROR].copy()

# Apply preprocessing
X_train_l3_processed = preprocessor_l3.fit_transform(X_train_l3)
X_test_l3_processed = preprocessor_l3.transform(X_test_l3)

print(f"Shape of preprocessed training data for L3 (Billing Error): {X_train_l3_processed.shape}")
print(f"Shape of preprocessed test data for L3 (Billing Error): {X_test_l3_processed.shape}")

# --- Random Forest Model for Is_Billing_Error ---
billing_error_ratio_train = y_train_l3_be.value_counts(normalize=True)
print(f"Billing Error ratio in L3 training data: \n{billing_error_ratio_train}")

# RandomForest can handle imbalance somewhat with class_weight='balanced'
rf_l3_be_model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42,
    class_weight='balanced', # Good for imbalanced classes
    n_jobs=-1, # Use all available cores
    verbose=0
)

print("\nTraining Layer 3 Random Forest model for Is_Billing_Error...")
rf_l3_be_model.fit(X_train_l3_processed, y_train_l3_be)
print("Layer 3 (Is_Billing_Error) model training complete.")

# --- Predictions and Evaluation for Is_Billing_Error ---
y_pred_l3_be_proba = rf_l3_be_model.predict_proba(X_test_l3_processed)[:, 1]
y_pred_l3_be_class = rf_l3_be_model.predict(X_test_l3_processed)

test_df['Layer3_Billing_Error_Probability'] = y_pred_l3_be_proba
test_df['Layer3_Billing_Error_Prediction'] = y_pred_l3_be_class

print("\nLayer 3 Is_Billing_Error Performance (on Test Set):")
print(classification_report(y_test_l3_be_actual, y_pred_l3_be_class, target_names=['Not Billing Error', 'Billing Error']))

try:
    roc_auc_l3_be = roc_auc_score(y_test_l3_be_actual, y_pred_l3_be_proba)
    print(f"Layer 3 (Is_Billing_Error) ROC AUC Score: {roc_auc_l3_be:.4f}")
except ValueError as e:
    print(f"Could not calculate ROC AUC for Layer 3 (Is_Billing_Error): {e}")
    roc_auc_l3_be = None

if roc_auc_l3_be is not None:
    fpr, tpr, _ = roc_curve(y_test_l3_be_actual, y_pred_l3_be_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Layer 3 RF (BE AUC = {roc_auc_l3_be:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Layer 3 ROC Curve - Is_Billing_Error')
    plt.legend(loc='lower right')
    plt.show()

# Confusion Matrix for Is_Billing_Error
cm_l3_be = confusion_matrix(y_test_l3_be_actual, y_pred_l3_be_class)
plt.figure(figsize=(7, 5))
sns.heatmap(cm_l3_be, annot=True, fmt='d', cmap='Greens', xticklabels=['Not BE', 'BE'], yticklabels=['Not BE', 'BE'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Layer 3 Confusion Matrix - Is_Billing_Error')
plt.show()


# --- Optional: Model for Billing_Error_Type (Multi-class) ---
# Only train this on actual billing errors to predict the type
train_df_be_only = train_df[train_df[TARGET_BILLING_ERROR] == 1].copy()
if not train_df_be_only.empty and 'Billing_Error_Type' in train_df_be_only.columns and train_df_be_only['Billing_Error_Type'].nunique() > 1:
    print("\n--- Training Layer 3 Model for Billing_Error_Type ---")
    X_train_l3_betype = train_df_be_only[features_l3].copy() # Use same features
    y_train_l3_betype = train_df_be_only['Billing_Error_Type'].copy().fillna('Unknown') # Fill NaNs for target

    # Preprocess
    X_train_l3_betype_processed = preprocessor_l3.transform(X_train_l3_betype) # Use already fitted preprocessor

    rf_l3_betype_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1,
        verbose=0
    )
    print("Training Layer 3 Random Forest model for Billing_Error_Type...")
    rf_l3_betype_model.fit(X_train_l3_betype_processed, y_train_l3_betype)
    print("Layer 3 (Billing_Error_Type) model training complete.")

    # Predict on test set transactions that were *predicted* as billing errors by the first L3 model
    # Or, for evaluation, predict on all test transactions and then filter by actual billing errors
    test_df_be_actual_only = test_df[test_df[TARGET_BILLING_ERROR] == 1].copy()
    if not test_df_be_actual_only.empty :
        X_test_l3_betype = test_df_be_actual_only[features_l3].copy()
        y_test_l3_betype_actual = test_df_be_actual_only['Billing_Error_Type'].copy().fillna('Unknown')

        X_test_l3_betype_processed = preprocessor_l3.transform(X_test_l3_betype)
        y_pred_l3_betype_class = rf_l3_betype_model.predict(X_test_l3_betype_processed)

        print("\nLayer 3 Billing_Error_Type Performance (on actual Billing Errors in Test Set):")
        print(classification_report(y_test_l3_betype_actual, y_pred_l3_betype_class))
        
        # Add this prediction to the main test_df for those records predicted as Billing Error by the BE model
        # For simplicity, we'll make a general prediction on all test_df and then use it if Layer3_Billing_Error_Prediction is 1
        all_X_test_l3_betype_processed = preprocessor_l3.transform(test_df[features_l3]) # Transform all test data
        all_y_pred_l3_betype_class = rf_l3_betype_model.predict(all_X_test_l3_betype_processed)
        test_df['Layer3_Predicted_Error_Type'] = all_y_pred_l3_betype_class
        # Only keep the predicted type if the BE model also flagged it as a billing error
        test_df.loc[test_df['Layer3_Billing_Error_Prediction'] == 0, 'Layer3_Predicted_Error_Type'] = None
    else:
        print("No actual billing errors in test set to evaluate Billing_Error_Type model.")
        test_df['Layer3_Predicted_Error_Type'] = None # No predictions to make

else:
    print("Skipping Billing_Error_Type model: not enough data or distinct types in training set.")
    test_df['Layer3_Predicted_Error_Type'] = None
