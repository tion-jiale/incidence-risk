import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import json


import xgboost as xgb
#from catboost import CatBoostClassifier

# ------------------------------------------------
# Load data
# ------------------------------------------------
csv_path = os.path.join(os.path.dirname(__file__), "final_data.csv")
final_data = pd.read_csv(csv_path)

X = final_data.drop(columns=["incidence_encoded", "incidence_quartile_custom"])
y = final_data["incidence_encoded"]

# ------------------------------------------------
# Train-test split
# ------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ------------------------------------------------
# SMOTE (train only)
# ------------------------------------------------
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# ------------------------------------------------
# Scaling (ONLY for Logistic Regression)
# ------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------
# Logistic Regression
# ------------------------------------------------
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train_sm)

# ------------------------------------------------
# Random Forest (NO scaling)
# ------------------------------------------------
rf_model = RandomForestClassifier(
    n_estimators=300,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
rf_model.fit(X_train_sm, y_train_sm)

# ------------------------------------------------
# XGBoost (NO scaling)
# ------------------------------------------------
xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="mlogloss",
    use_label_encoder=False,
    random_state=42
)
xgb_model.fit(X_train_sm, y_train_sm)

# ------------------------------------------------
# CatBoost (NO scaling)
# ------------------------------------------------
#cb_model = CatBoostClassifier(
    #iterations=300,
    #depth=6,
    #learning_rate=0.1,
    #loss_function="MultiClass",
    #verbose=False,
    #random_seed=42
#)
#cb_model.fit(X_train_sm, y_train_sm)


# ------------------------------------------------
# Model Evaluation (Test Set)
# ------------------------------------------------
metrics = {}

# Random Forest
rf_pred = rf_model.predict(X_test)
rf_proba = rf_model.predict_proba(X_test)

metrics["Random Forest"] = {
    "Accuracy": (rf_pred == y_test).mean(),
    "Precision": precision_score(y_test, rf_pred, average="macro"),
    "Recall": recall_score(y_test, rf_pred, average="macro"),
    "F1-score": f1_score(y_test, rf_pred, average="macro"),
    "ROC-AUC": roc_auc_score(y_test, rf_proba, multi_class="ovr")
}

# Logistic Regression (scaled)
X_test_scaled = scaler.transform(X_test)
lr_pred = lr.predict(X_test_scaled)
lr_proba = lr.predict_proba(X_test_scaled)

metrics["Logistic Regression"] = {
    "Accuracy": (lr_pred == y_test).mean(),
    "Precision": precision_score(y_test, lr_pred, average="macro"),
    "Recall": recall_score(y_test, lr_pred, average="macro"),
    "F1-score": f1_score(y_test, lr_pred, average="macro"),
    "ROC-AUC": roc_auc_score(y_test, lr_proba, multi_class="ovr")
}

# XGBoost
xgb_pred = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)

metrics["XGBoost"] = {
    "Accuracy": (xgb_pred == y_test).mean(),
    "Precision": precision_score(y_test, xgb_pred, average="macro"),
    "Recall": recall_score(y_test, xgb_pred, average="macro"),
    "F1-score": f1_score(y_test, xgb_pred, average="macro"),
    "ROC-AUC": roc_auc_score(y_test, xgb_proba, multi_class="ovr")
}

# CatBoost
#cb_pred = cb_model.predict(X_test)
#cb_proba = cb_model.predict_proba(X_test)

#metrics["CatBoost"] = {
 #   "Precision": precision_score(y_test, cb_pred, average="macro"),
  #  "Recall": recall_score(y_test, cb_pred, average="macro"),
  #  "F1-score": f1_score(y_test, cb_pred, average="macro"),
  #  "ROC-AUC": roc_auc_score(y_test, cb_proba, multi_class="ovr")
#}
with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("✅ Evaluation metrics saved")

# ------------------------------------------------
# Save artifacts
# ------------------------------------------------
joblib.dump(lr, "logistic_regression_model.pkl")
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(xgb_model, "xgboost_model.pkl")
# joblib.dump(cb_model, "catboost_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("✅ Models, scaler, and columns saved successfully")

