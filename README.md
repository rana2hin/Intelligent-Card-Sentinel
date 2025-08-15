# Intelligent Credit Sentinel 🛡️

A multi-layer pipeline for credit card **fraud** and **billing error**
detection.  The system values **recall** – catching every suspicious
transaction – while providing interpretable scores and visual feedback.

## Data & Pre‑processing
`data_utils.load_datasets` reads the provided
`simulated_credit_card_transactions.csv`, sorts records by timestamp and
creates chronological train/validation/test splits.  Numeric features are
scaled, categoricals are one‑hot encoded and velocity statistics are
automatically generated.  Training splits can be oversampled to balance rare
classes using `RandomOverSampler`.

Run the helper script to inspect the splits:

```bash
python "Load Data and Initial Preprocessing.py"
```

## Layer 1 – Rapid Anomaly Screener 🔍
Unsupervised auto‑encoder trained only on *normal* transactions.  Reconstruction
error provides an anomaly score.  Optuna tunes the architecture to maximise the
precision‑recall AUC on a validation set.  The layer outputs:

- `Layer1_Reconstruction_Error`
- `Layer1_Is_Anomaly_Flag`

## Layer 2 – Fraud Likelihood Estimator ⚠️
Supervised XGBoost classifier predicting `Is_Fraud`.  The oversampled training
set and hyper‑parameter search (recall scoring) deliver high sensitivity.  Key
plots include ROC and precision‑recall curves plus confusion matrices.

Outputs:
- `Layer2_Fraud_Probability`
- `Layer2_Fraud_Prediction`

## Layer 3 – Billing Anomaly Detector 🧾
Balanced Random Forest classifying `Is_Billing_Error`.  Like Layer 2 the model
uses oversampling and recall‑oriented tuning.  Evaluation mirrors Layer 2 and
helps highlight duplicate charges or subscription issues.

Outputs:
- `Layer3_Billing_Error_Probability`
- `Layer3_Billing_Error_Prediction`

## Layer 4 – Decision & Action Engine 🤖
A lightweight logistic‑regression meta learner combines the previous layers.
The final risk score triggers suggested actions (approve, review, decline)
based on configurable thresholds.  This stage illustrates how model outputs can
inform downstream business rules.

## Running the Pipeline
Each layer is an independent script; run them sequentially after data loading.

```bash
python "Layer 1- Rapid Anomaly Screener (Autoencoder).py"
python "Layer 2- Fraud Likelihood Estimator.py"
python "Layer 3- Billing Anomaly Detector.py"
# After the above three have populated predictions:
python "Layer 4- The Decision & Action Engine.py"
```

## Evaluation
All layers report accuracy, precision, recall and F1 scores as well as visual
curves.  Since financial loss from missed fraud is high, the models are tuned
for **maximum recall** with precision traded off where necessary.

## License
This project is released under the MIT License.

