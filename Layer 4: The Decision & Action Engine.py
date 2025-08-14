print("\n--- Layer 4: Decision & Action Engine (Meta-Learner Conception) ---")

# For this demonstration, we'll split our current test_df further
# to get a "meta_train_df" (for training the meta-learner) and a "meta_test_df" (for evaluating it)
# This simulates having a validation set with predictions from L1, L2, L3.

if 'Layer1_Reconstruction_Error' not in test_df.columns or \
   'Layer2_Fraud_Probability' not in test_df.columns or \
   'Layer3_Billing_Error_Probability' not in test_df.columns:
    print("Error: Outputs from Layers 1, 2, or 3 are missing in test_df. Cannot proceed with Layer 4.")
    # In a real pipeline, ensure these are populated.
    # For now, if they are missing, we'll create dummy columns for the code to run.
    if 'Layer1_Reconstruction_Error' not in test_df.columns: test_df['Layer1_Reconstruction_Error'] = np.random.rand(len(test_df))
    if 'Layer2_Fraud_Probability' not in test_df.columns: test_df['Layer2_Fraud_Probability'] = np.random.rand(len(test_df))
    if 'Layer3_Billing_Error_Probability' not in test_df.columns: test_df['Layer3_Billing_Error_Probability'] = np.random.rand(len(test_df))
    print("Dummy L1/L2/L3 outputs created in test_df for Layer 4 demonstration.")


meta_split_point = int(len(test_df) * 0.5) # Use 50% of test_df for meta-training
meta_train_df = test_df.iloc[:meta_split_point].copy()
meta_test_df = test_df.iloc[meta_split_point:].copy()

print(f"Meta-Train set shape: {meta_train_df.shape}")
print(f"Meta-Test set shape: {meta_test_df.shape}")

# --- Define Features and Target for Meta-Learner ---
meta_features = [
    'Layer1_Reconstruction_Error',
    'Layer2_Fraud_Probability',
    'Layer3_Billing_Error_Probability',
    'Transaction_Amount_Local_Currency' # Example: include original transaction amount
]
# Ensure features exist
meta_features = [f for f in meta_features if f in meta_train_df.columns]


# Define a simplified combined target: "High_Risk_Event"
# 1 if it's fraud OR a significant billing error (e.g., probability > 0.5 from L3)
# This is a simplification for demonstration.
meta_train_df['Meta_Target_High_Risk'] = (
    (meta_train_df[TARGET_FRAUD] == 1) |
    (meta_train_df[TARGET_BILLING_ERROR] == 1) # Simplified: any billing error is high risk for this target
).astype(int)

meta_test_df['Meta_Target_High_Risk'] = (
    (meta_test_df[TARGET_FRAUD] == 1) |
    (meta_test_df[TARGET_BILLING_ERROR] == 1)
).astype(int)

X_meta_train = meta_train_df[meta_features].copy()
y_meta_train = meta_train_df['Meta_Target_High_Risk'].copy()

X_meta_test = meta_test_df[meta_features].copy()
y_meta_test_actual = meta_test_df['Meta_Target_High_Risk'].copy()

# --- Preprocessing for Meta-Learner (usually just scaling) ---
# Impute NaNs that might arise from layer predictions (though ideally they shouldn't)
meta_preprocessor = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

X_meta_train_processed = meta_preprocessor.fit_transform(X_meta_train)
X_meta_test_processed = meta_preprocessor.transform(X_meta_test)

print(f"Shape of preprocessed meta-training data: {X_meta_train_processed.shape}")

# --- Meta-Learner Model (Logistic Regression) ---
meta_learner = LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced')

print("\nTraining Layer 4 Meta-Learner...")
meta_learner.fit(X_meta_train_processed, y_meta_train)
print("Layer 4 Meta-Learner training complete.")

# --- Predictions and Evaluation for Meta-Learner ---
y_pred_meta_proba = meta_learner.predict_proba(X_meta_test_processed)[:, 1]
y_pred_meta_class = meta_learner.predict(X_meta_test_processed)

meta_test_df['Layer4_Final_Risk_Probability'] = y_pred_meta_proba
meta_test_df['Layer4_Final_Risk_Prediction'] = y_pred_meta_class

print("\nLayer 4 Meta-Learner Performance (on Meta-Test Set for 'High_Risk_Event'):")
print(classification_report(y_meta_test_actual, y_pred_meta_class, target_names=['Not High Risk', 'High Risk']))

try:
    roc_auc_meta = roc_auc_score(y_meta_test_actual, y_pred_meta_proba)
    print(f"Layer 4 Meta-Learner ROC AUC Score: {roc_auc_meta:.4f}")
except ValueError as e:
    print(f"Could not calculate ROC AUC for Layer 4: {e}")
    roc_auc_meta = None
    
# --- Example of how to use the meta-learner's output for actions ---
# This is where you'd define thresholds for actions
final_decision_threshold_high_risk = 0.7 # Example: if prob > 0.7, take strong action
final_decision_threshold_medium_risk = 0.4 # Example: if prob > 0.4, flag for review

meta_test_df['Suggested_Action'] = 'Approve'
meta_test_df.loc[meta_test_df['Layer4_Final_Risk_Probability'] > final_decision_threshold_medium_risk, 'Suggested_Action'] = 'Flag_For_Review'
meta_test_df.loc[meta_test_df['Layer4_Final_Risk_Probability'] > final_decision_threshold_high_risk, 'Suggested_Action'] = 'Decline_Or_StepUp'

print("\nExample Suggested Actions based on Layer 4 Meta-Learner Output:")
print(meta_test_df[['Layer1_Reconstruction_Error', 'Layer2_Fraud_Probability', 'Layer3_Billing_Error_Probability',
                    TARGET_FRAUD, TARGET_BILLING_ERROR, # Actuals
                    'Layer4_Final_Risk_Probability', 'Suggested_Action']].head(20))

print("\nDistribution of Suggested Actions:")
print(meta_test_df['Suggested_Action'].value_counts())

import matplotlib.pyplot as plt

# Calculate counts directly from the DataFrame column
counts = meta_test_df['Suggested_Action'].value_counts()

# Plot the bar chart
plt.figure(figsize=(8, 6))
counts.plot(kind='bar')
plt.title("Distribution of Suggested Actions")
plt.xlabel("Suggested Action")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
