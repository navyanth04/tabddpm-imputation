import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
args = parser.parse_args()

DATASET = args.dataset
# -----------------------------
# CHANGE THESE
# -----------------------------
IMPUTED_PATH = f"exp/{DATASET}/tabddpm_impute/imputed"
OUT_FILE = f"exp/{DATASET}/tabddpm_impute/downstream_metrics.json"

# -----------------------------
# LOAD DATA
# -----------------------------
X_train = np.load(os.path.join(IMPUTED_PATH, "X_num_train.npy"))
X_val   = np.load(os.path.join(IMPUTED_PATH, "X_num_val.npy"))
X_test  = np.load(os.path.join(IMPUTED_PATH, "X_num_test.npy"))

y_train = np.load(os.path.join(IMPUTED_PATH, "y_train.npy"))
y_val   = np.load(os.path.join(IMPUTED_PATH, "y_val.npy"))
y_test  = np.load(os.path.join(IMPUTED_PATH, "y_test.npy"))

# combine train + val for final model fitting
X_tr = np.concatenate([X_train, X_val], axis=0)
y_tr = np.concatenate([y_train, y_val], axis=0)

# -----------------------------
# MODELS
# -----------------------------
models = {
    "xgboost": XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    ),
    "logistic_regression": LogisticRegression(
        max_iter=2000,
        random_state=42
    )
}

results = {}

for name, model in models.items():
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred))
    }

    if y_prob is not None:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_prob))

    results[name] = metrics

with open(OUT_FILE, "w") as f:
    json.dump(results, f, indent=4)

print("Saved downstream metrics to:", OUT_FILE)
print(json.dumps(results, indent=2))
