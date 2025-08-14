# Inteligent Credit Sentinel
## Overall Structure:

1.  **Load and Preprocess Data:** Common steps for all layers.
2.  **Layer 1: Rapid Anomaly Screener (Autoencoder)**
3.  **Layer 2: Fraud Likelihood Estimator (Gradient Boosting)**
4.  **Layer 3: Billing Anomaly Detector (Random Forest / Logistic Regression with NLP hints)**
5.  **Layer 4: Decision & Action Engine (Conceptual - Meta-Learner)**

We'll start with loading, preprocessing, and then Layer 1.

**Important Considerations Before Coding:**

*   **Feature Selection:** For each layer, we'll select relevant features from the simulated dataset. Not all generated features are useful for every layer.
*   **Train-Test Split:** Crucial for evaluating model performance. We should ideally split based on time if we want to simulate a real deployment, but for this initial prototype, a random split is acceptable. Let's use a time-based split as it's more robust.
*   **Evaluation Metrics:**
    *   **Anomaly Detection (Layer 1):** Reconstruction error distribution, AUC-ROC if we treat high reconstruction error as "positive class."
    *   **Classification (Layers 2 & 3):** Precision, Recall, F1-Score, AUC-ROC, Confusion Matrix. Particularly important for imbalanced classes (fraud, billing errors).
*   **Simplicity First:** We'll start with standard implementations. Hyperparameter tuning and more complex architectures can come later.

## Part 1: Load Data and Initial Preprocessing

## Part 2: Layer 1 - Rapid Anomaly Screener (Autoencoder)
The goal here is to train an autoencoder on normal (non-fraud, non-billing_error) transactions from the training set. Then, we'll use it to get reconstruction errors for all transactions in the test set. High reconstruction error suggests an anomaly.

**Explanation of Layer 1 Code:**

1.  **Feature Selection:** We pick features relevant for general anomaly detection – core transaction details and how they deviate from the cardholder's norm.
2.  **Preprocessor (`preprocessor_l1_minmax`):**
    *   `MinMaxScaler` is used for numerical features because autoencoders with sigmoid activation in the output layer typically expect inputs scaled between 0 and 1.
    *   `OneHotEncoder` converts categorical features into numerical format.
3.  **Data Preparation:**
    *   The autoencoder is trained *only* on transactions from `train_df` that are marked as `Is_Fraud == 0` AND `Is_Billing_Error == 0`. This teaches the autoencoder what "normal" looks like.
4.  **Autoencoder Architecture:**
    *   A simple feedforward neural network with an "encoder" part (reducing dimensionality) and a "decoder" part (reconstructing the input).
    *   `relu` activation is common in hidden layers.
    *   `sigmoid` activation in the output layer matches the 0-1 scaled input.
    *   `mse` (Mean Squared Error) is used as the loss function, aiming to minimize the difference between input and reconstructed output.
5.  **Training:**
    *   `EarlyStopping` is used to prevent overfitting and stop training when validation loss stops improving.
6.  **Reconstruction Error:**
    *   After training, the autoencoder is used to reconstruct all transactions in the `test_df`.
    *   The MSE between the original (scaled) test data and its reconstruction is calculated for each transaction. This is our anomaly score.
7.  **Evaluation:**
    *   Histograms and boxplots help visualize the distribution of reconstruction errors and see if they differ for fraudulent/billing error transactions.
8.  **Anomaly Threshold:** A common way to flag anomalies is to set a threshold on the reconstruction error. We calculate the 95th percentile of reconstruction errors on the *normal training data* and use that as a threshold. Transactions in the test set with errors above this threshold are flagged as anomalous by Layer 1.
9.  **Output:** The `Layer1_Reconstruction_Error` and `Layer1_Is_Anomaly_Flag` are added to `test_df`. These will be important inputs/features for later layers or the final decision engine.

---

Next, we'll move to Layer 2 (Fraud Likelihood Estimator). This layer will use supervised learning.

## Part 3: Layer 2- Fraud Likelihood Estimator.
This layer takes transactions (potentially all, or those flagged as anomalous by Layer 1, though for a robust system, it's often better to score all for Layer 2) and predicts the probability of them being fraudulent. This is a supervised classification task. We'll use GradientBoostingClassifier as it's generally powerful for tabular data.

**Explanation of Layer 2 Code:**

1.  **Feature Selection:**
    *   We select a broader set of features than Layer 1, including those indicative of specific fraud patterns (velocity, merchant risk, CNP details if they were more richly simulated).
    *   The `Layer1_Reconstruction_Error` *could* be a feature here if we had run Layer 1 on the training data too. For simplicity in this pass, Layer 2 is trained independently. In a real system, outputs of previous layers (on training folds) are often inputs to subsequent layers.
2.  **Preprocessor (`preprocessor_l2`):**
    *   Uses `StandardScaler` for numerical features (common for tree-based models, though not strictly necessary, it can sometimes help).
    *   `OneHotEncoder` for categorical features.
    *   `remainder='passthrough'` or `'drop'` can be chosen. If `passthrough`, ensure non-transformed columns are handled or removed before `fit`. For this example, we explicitly select features for `X_train_l2` and `X_test_l2` to match what the preprocessor expects.
3.  **Data Preparation:**
    *   `X_train_l2` and `X_test_l2` are prepared using the selected features.
    *   `y_train_l2` is the `Is_Fraud` column.
4.  **Gradient Boosting Model (`gbt_l2_model`):**
    *   `GradientBoostingClassifier` from `sklearn.ensemble` is used.
    *   Basic hyperparameters like `n_estimators`, `learning_rate`, `max_depth`, and `subsample` are set. These would be tuned in a real project (e.g., using GridSearchCV or RandomizedSearchCV).
    *   **Class Imbalance:** The code checks the fraud ratio. If it's highly imbalanced, Gradient Boosting might need strategies like:
        *   Using `sample_weight` in the `.fit()` method if you can calculate appropriate weights.
        *   Using libraries like `imbalanced-learn` for oversampling (e.g., SMOTE) or undersampling.
        *   Using a different model more robust to imbalance (like `XGBoost` with `scale_pos_weight`). For this initial pass, we're keeping it simple.
5.  **Training:** The model is trained on the preprocessed training data.
6.  **Prediction and Evaluation:**
    *   `predict_proba` gives the probability of each class (we take the probability of the positive class, i.e., fraud).
    *   `predict` gives the direct class prediction (0 or 1).
    *   **Metrics:** `classification_report` (precision, recall, F1-score), `roc_auc_score`, ROC curve, Precision-Recall curve, and Confusion Matrix are used to evaluate performance on the *test set*. These are critical for understanding how well the model identifies fraud and manages false positives/negatives.
7.  **Feature Importance:** Gradient Boosting models can provide feature importances, showing which features contributed most to the predictions. This is valuable for understanding the model and for potential feature selection.
8.  **Output:** The `Layer2_Fraud_Probability` and `Layer2_Fraud_Prediction` are added to `test_df`. The probability score is especially useful for Layer 4 (Decision Engine) as it allows for flexible thresholding.

This provides a solid fraud detection model for Layer 2. Next up will be Layer 3 for Billing Anomaly Detection.

## Part 4: Layer 3- Billing Anomaly Detector.
This layer focuses on identifying non-fraudulent but potentially erroneous or disputable charges (like duplicates or unwanted subscriptions). It's also a supervised classification task, but the features and potentially the model choice might differ slightly from Layer 2. We'll target `Is_Billing_Error` and try to predict `Billing_Error_Type`.

For this layer, features related to transaction descriptors, merchant billing history, and comparison with recent transactions become more important. Since our simulated `Transaction_Descriptor_Raw_Text` is not richly populated by Faker for deep NLP, we'll rely more on the structured comparison features and flags we generated. In a real scenario, NLP on detailed descriptors would be key here.

We can use a `RandomForestClassifier` or `LogisticRegression` as they are good general-purpose classifiers. Let's go with RandomForest as it handles categorical features well (after encoding) and can capture non-linearities.

**Explanation of Layer 3 Code:**

1.  **Feature Selection:**
    *   We select features that are more likely to indicate billing issues. This includes `Time_Since_CH_Last_Transaction_at_Same_Merchant_Min` (key for duplicates), `Historical_Billing_Dispute_Rate_Global`, and general cardholder/transaction context.
    *   Ideally, features derived from `Transaction_Descriptor_Raw_Text` (e.g., presence of keywords like "TRIAL," "RECURRING," "AUTORENEW") and flags like `Is_Recurring_Payment_Flag` would be very powerful here. Our simulation is basic in this aspect, so the model will rely more on other patterns.
2.  **Preprocessor (`preprocessor_l3`):** Similar to Layer 2, using `StandardScaler` and `OneHotEncoder`.
3.  **Data Preparation:**
    *   The primary target is `Is_Billing_Error`.
    *   `X_train_l3` and `X_test_l3` are prepared.
4.  **Random Forest Model for `Is_Billing_Error` (`rf_l3_be_model`):**
    *   `RandomForestClassifier` is chosen.
    *   `class_weight='balanced'` is used to help address the imbalance in the `Is_Billing_Error` target.
5.  **Training and Evaluation:**
    *   The model is trained to predict whether a transaction is a billing error.
    *   Standard classification metrics (report, ROC AUC, confusion matrix) are used for evaluation.
6.  **Optional: Model for `Billing_Error_Type`:**
    *   A second RandomForest model (`rf_l3_betype_model`) is trained *only on transactions that are actual billing errors* in the training set (`train_df_be_only`).
    *   Its goal is to classify the *type* of billing error (e.g., `Duplicate_Charge`, `Unwanted_Subscription_Renewal`).
    *   This model's predictions are then applied to the test set. We store `Layer3_Predicted_Error_Type` in `test_df`, but it's only meaningful if `Layer3_Billing_Error_Prediction` is also 1.
7.  **Output:** `Layer3_Billing_Error_Probability`, `Layer3_Billing_Error_Prediction`, and `Layer3_Predicted_Error_Type` are added to `test_df`. These provide nuanced information about potential billing issues.

This completes the modeling for the primary detection layers. The final step is conceptualizing Layer 4 (Decision & Action Engine), which would use the outputs of Layers 1, 2, and 3.

## Part 5: Layer 4: The "Decision & Action Engine".
This layer isn't typically a single, complex ML model trained from scratch like the previous ones. Instead, it's a system that **integrates the outputs (scores/probabilities) from Layers 1, 2, and 3** to make a final, informed decision and suggest an appropriate action.

**Objectives of Layer 4:**

1.  **Synthesize Information:** Combine potentially conflicting or complementary signals from the specialized upstream layers.
2.  **Apply Business Logic:** Incorporate bank-specific rules, risk appetite, and customer segmentation.
3.  **Dynamic Thresholding:** Allow for adjustment of decision boundaries based on current risk levels or specific scenarios.
4.  **Action Prioritization:** Determine the most appropriate response (e.g., decline, alert, flag for review, step-up authentication).

**Approaches for Layer 4:**

1.  **Rule-Based System:**
    *   **How it works:** A set of predefined `IF-THEN-ELSE` rules based on the scores from Layers 1, 2, and 3, and other contextual variables (e.g., transaction amount, customer value).
    *   **Example Rules:**
        *   `IF Layer2_Fraud_Probability > 0.9 THEN DeclineTransaction AND AlertUser`
        *   `IF Layer2_Fraud_Probability > 0.6 AND Layer1_Reconstruction_Error > THRESHOLD_HIGH THEN StepUpAuthentication`
        *   `IF Layer3_Billing_Error_Probability > 0.8 AND Layer2_Fraud_Probability < 0.2 THEN FlagForCustomerReview ("Does this look right?")`
        *   `IF Layer3_Predicted_Error_Type == 'Duplicate_Charge' AND Layer3_Billing_Error_Probability > 0.7 THEN FlagForAnalystReview`
    *   **Pros:** Transparent, easy to understand and modify by business users.
    *   **Cons:** Can become very complex to manage, may not capture subtle interactions between scores, requires manual tuning.

2.  **Weighted Scoring System:**
    *   **How it works:** Assign weights to the scores from each layer and combine them into a final risk score.
    *   **Example:** `Final_Risk_Score = (w1 * Layer1_Anomaly_Score_Normalized) + (w2 * Layer2_Fraud_Probability) + (w3 * Layer3_Billing_Error_Probability)`
    *   Different thresholds on `Final_Risk_Score` trigger different actions.
    *   **Pros:** Simpler than extensive rule sets.
    *   **Cons:** Determining optimal weights can be challenging, might oversimplify.

3.  **Meta-Learner (Stacking):**
    *   **How it works:** Train a simple machine learning model (the meta-learner, e.g., Logistic Regression, a shallow Decision Tree, or even a simple Neural Network) using the outputs (probabilities/scores) from Layers 1, 2, and 3 as its input features. The target for this meta-learner would be the ultimate desired outcome (e.g., a combined "action code" or a refined "final risk level").
    *   **Data for Meta-Learner:**
        *   **Features:** `Layer1_Reconstruction_Error`, `Layer2_Fraud_Probability`, `Layer3_Billing_Error_Probability`. You could also include the original transaction amount or other key context.
        *   **Target:** This is the trickiest part. You need a "ground truth" for the *optimal combined decision*. This might come from:
            *   Historical data where human analysts made final decisions based on various alerts.
            *   Simulating optimal actions based on the known `Is_Fraud` and `Is_Billing_Error` flags and a predefined cost matrix (e.g., cost of a false positive vs. cost of a missed fraud).
    *   **Pros:** Can learn optimal ways to combine the signals from previous layers, potentially capturing non-linear interactions.
    *   **Cons:** Requires careful setup of training data and target definition, can be less transparent than rules.

**Let's implement a simple Meta-Learner (Logistic Regression) for demonstration.**

**Assumptions for the Meta-Learner:**

*   We'll create a simplified target for the meta-learner. Let's define a combined "High_Risk_Event" if either `Is_Fraud` is true OR `Is_Billing_Error` is true (and it's a significant billing error). This is a simplification; a real system would have more nuanced action targets.
*   We need to generate predictions from Layers 1, 2, and 3 on a *validation set* (or use cross-validation predictions on the training set) to train the meta-learner. Using predictions from models on the same data they were trained on to train a meta-learner leads to leakage and overly optimistic results.

**Since we've already added Layer 1, 2, and 3 predictions to our `test_df`, we can use a portion of this `test_df` as a "holdout validation set" to train the meta-learner, and then evaluate on the remainder of `test_df`. This isn't ideal (ideally, we'd have a separate validation set from the start), but it's feasible for this demonstration.**

**Explanation of Layer 4 (Meta-Learner) Code:**

1.  **Data Split:** The existing `test_df` (which now contains predictions from L1, L2, L3) is further split. `meta_train_df` is used to train the meta-learner, and `meta_test_df` is used to evaluate it. This ensures the meta-learner is trained on "out-of-sample" predictions from the base layers.
2.  **Feature Selection:** The inputs (`meta_features`) to the meta-learner are the key outputs from the previous layers:
    *   `Layer1_Reconstruction_Error`
    *   `Layer2_Fraud_Probability`
    *   `Layer3_Billing_Error_Probability`
    *   We also included `Transaction_Amount_Local_Currency` as an example of how other raw features can provide context to the meta-learner.
3.  **Target Definition (`Meta_Target_High_Risk`):**
    *   This is crucial and often the most complex part to define for a meta-learner.
    *   For this demonstration, we created a simple binary target: `1` if the transaction was *actually* fraudulent (`Is_Fraud == 1`) OR if it was *actually* a billing error (`Is_Billing_Error == 1`). This treats both as a "high-risk event" we want the meta-learner to identify.
    *   In a real system, you might have multiple target classes representing different actions (e.g., 0=Approve, 1=Flag, 2=Step-Up, 3=Decline) and train a multi-class meta-learner, or have separate meta-learners for different types of risk.
4.  **Preprocessing:** The input features to the meta-learner are scaled using `StandardScaler`.
5.  **Meta-Learner Model:** A `LogisticRegression` model is used. It's simple, interpretable, and often works well as a meta-learner. `class_weight='balanced'` is used as the `Meta_Target_High_Risk` might also be imbalanced.
6.  **Training and Evaluation:**
    *   The meta-learner is trained on `X_meta_train_processed` and `y_meta_train`.
    *   It's evaluated on `X_meta_test_processed` against `y_meta_test_actual`.
    *   Standard classification metrics are reported.
7.  **Suggested Actions:**
    *   The code demonstrates how the `Layer4_Final_Risk_Probability` from the meta-learner can be used with simple thresholds to determine a `Suggested_Action`. This is where the business logic and risk appetite of the bank would heavily influence the thresholds.

**Next Steps and Considerations for a Real System:**

*   **Proper Stacking Implementation:** For robust stacking, predictions for the meta-learner's training data should come from cross-validation on the original training set (to avoid leakage). Base models (L1, L2, L3) are trained on K-1 folds, predict on the Kth fold, and these out-of-fold predictions form the features for the meta-learner. Then, the base models are retrained on all training data to make predictions on the final test set.
*   **More Sophisticated Target for Meta-Learner:** Consider a cost-benefit analysis to define the target variable or use human analyst decisions as the ground truth.
*   **Dynamic Thresholding:** Thresholds for `Suggested_Action` should not be static. They could be adjusted based on:
    *   Overall fraud trends.
    *   Customer segments (e.g., higher tolerance for high-value customers).
    *   Specific merchant risk.
*   **Explainability (XAI):** For a system making critical financial decisions, understanding *why* a decision was made is important (e.g., using SHAP values for the meta-learner or even the base learners).
*   **Feedback Loop:** The actual outcomes of the `Suggested_Action` (was it correct? did the customer confirm fraud? was the billing error resolved?) are vital for retraining ALL layers of the system.

This multi-layered approach provides a powerful and flexible framework. We've now prototyped the core ML components for each detection stage and a way to combine their insights!
