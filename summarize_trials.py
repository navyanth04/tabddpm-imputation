import os
import json
import pandas as pd

RATES = [10, 20, 30, 40, 50]
TRIALS = [1, 2, 3, 4, 5]

rows = []

for rate in RATES:
    for trial in TRIALS:
        dataset = f"dataset_mcar_{rate}_trial{trial}"

        imp_file = f"exp/{dataset}/tabddpm_impute/eval_imputation/imputation_metrics.json"
        down_file = f"exp/{dataset}/tabddpm_impute/downstream_metrics.json"

        row = {
            "dataset": dataset,
            "missing_rate": rate,
            "trial": trial,
        }

        # -------------------
        # imputation metrics
        # -------------------
        if os.path.exists(imp_file):
            with open(imp_file, "r") as f:
                imp = json.load(f)

            test_overall = imp.get("test", {}).get("overall", {})
            val_overall = imp.get("val", {}).get("overall", {})
            train_overall = imp.get("train", {}).get("overall", {})

            row["train_rmse"] = train_overall.get("rmse")
            row["train_mae"] = train_overall.get("mae")
            row["train_kl"] = train_overall.get("kl")

            row["val_rmse"] = val_overall.get("rmse")
            row["val_mae"] = val_overall.get("mae")
            row["val_kl"] = val_overall.get("kl")

            row["test_rmse"] = test_overall.get("rmse")
            row["test_mae"] = test_overall.get("mae")
            row["test_kl"] = test_overall.get("kl")

        # -------------------
        # downstream metrics
        # -------------------
        if os.path.exists(down_file):
            with open(down_file, "r") as f:
                down = json.load(f)

            # adjust model name if needed
            for model_name in ["XGBoost", "RandomForest", "LogisticRegression"]:
                if model_name in down:
                    model_block = down[model_name]

                    # case 1: metrics under test
                    if "test" in model_block:
                        test_block = model_block["test"]
                    else:
                        test_block = model_block

                    row[f"{model_name}_accuracy"] = test_block.get("accuracy")
                    row[f"{model_name}_f1"] = test_block.get("f1")
                    row[f"{model_name}_roc_auc"] = test_block.get("roc_auc")

        rows.append(row)

df = pd.DataFrame(rows)
df = df.sort_values(["missing_rate", "trial"])

# save trial-wise table
os.makedirs("summary_results", exist_ok=True)
df.to_csv("summary_results/all_trials_metrics.csv", index=False)

# average across trials
avg = df.groupby("missing_rate").mean(numeric_only=True).reset_index()
std = df.groupby("missing_rate").std(numeric_only=True).reset_index()

avg.to_csv("summary_results/avg_by_missing_rate.csv", index=False)
std.to_csv("summary_results/std_by_missing_rate.csv", index=False)

print("Saved:")
print("  summary_results/all_trials_metrics.csv")
print("  summary_results/avg_by_missing_rate.csv")
print("  summary_results/std_by_missing_rate.csv")
