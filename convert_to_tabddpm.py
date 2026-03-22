import os
import numpy as np
import pandas as pd

base_path = "data"

cat_cols = ["code_module", "code_presentation"]
y_col = "y"

datasets = [
    d for d in os.listdir(base_path)
    if d.startswith("dataset_mcar_")
]

print(f"Found {len(datasets)} MCAR datasets")


def safe_read(path):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


for dataset in datasets:
    dataset_path = os.path.join(base_path, dataset)

    csv_file = os.path.join(dataset_path, f"{dataset}.csv")

    if not os.path.exists(csv_file):
        print(f"Skipping {dataset} (missing CSV)")
        continue

    print(f"\nProcessing {dataset}")

    df = safe_read(csv_file)

    # -----------------------------
    # ONLY DROP IF TARGET MISSING
    # -----------------------------
    if y_col not in df.columns:
        print(f"Skipping {dataset} (missing target column y)")
        continue

    df = df.dropna(subset=[y_col])   # ONLY THIS IS ALLOWED

    all_cols = df.columns.tolist()

    num_cols = [c for c in all_cols if c not in cat_cols + [y_col]]

    # -----------------------------
    # IMPORTANT CHANGE: NO DROPNA()
    # -----------------------------
    before = len(df)

    after = len(df)
    print(f"Rows kept: {after}")

    # -----------------------------
    # HANDLE MISSING VALUES SAFELY
    # -----------------------------
    X_num = df[num_cols].values.astype(np.float32)

    # replace NaNs in categorical with string token
    X_cat = df[cat_cols].fillna("missing").values.astype(str)

    y = df[y_col].values

    # -----------------------------
    # SPLITS
    # -----------------------------
    n = len(df)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    splits = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n)
    }

    for split, (s, e) in splits.items():
        np.save(os.path.join(dataset_path, f"X_num_{split}.npy"), X_num[s:e])
        np.save(os.path.join(dataset_path, f"X_cat_{split}.npy"), X_cat[s:e])
        np.save(os.path.join(dataset_path, f"y_{split}.npy"), y[s:e])

        print(f"{dataset}-{split}: {X_num[s:e].shape}, {X_cat[s:e].shape}, {y[s:e].shape}")

print("\nDONE")
