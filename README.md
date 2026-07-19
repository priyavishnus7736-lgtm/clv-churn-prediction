# Customer Lifetime Value (CLV) & Churn Prediction System

An end-to-end Data Science case study designed to analyze multi-source subscription customer data, engineer predictive behavior parameters, and optimize retention strategy matrices.

## 📊 Relational Data Ecosystem
The project combines information across 6 core data sheets:
* `customers.csv` - Account configurations and baseline demographic parameters.
* `transactions.csv` - Direct history tracking dates, balances, and refund flags.
* `events.csv` - Platform engagement counts and log metrics.
* `support_tickets.csv` - Resolution timelines and user satisfaction values.
* `churn_labels.csv` & `clv_labels.csv` - Binary prediction targets and numeric financial values.

## ⚙️ Implemented Pipelines
1. **Feature Engineering Module:** Builds operational RFM profiles (Recency, Frequency, Monetary), tenure periods, and customer support metrics.
2. **CLV Prediction Track (Regression):** Implements model learning targets checked via RMSE, MAE, and $R^2$.
3. **Churn Threat Track (Classification):** Addresses operational class imbalances and optimizes decisions based on AUC-ROC and F1-Scores.
