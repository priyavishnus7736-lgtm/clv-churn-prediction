import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, f1_score

def train_clv_regressor(X, y):
    """Evaluates Regression parameters for Customer Lifetime Value."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"CLV Regression R2 Score: {r2_score(y_test, preds):.4f}")
    return model

def train_churn_classifier(X, y):
    """Evaluates Classification metrics for Churn risk probabilities."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"Churn Classification AUC-ROC: {roc_auc_score(y_test, probs):.4f}")
    return model
