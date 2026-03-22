import os
import json
import numpy as np
import pandas as pd

BASE_PATH = "data"
ORIGINAL_CSV = os.path.join(BASE_PATH, "original_complete", "oula_complete.csv")
Y_COL = "y"

LOW_CARD_CAT_COLS = ["code_module", "code_presentation"]
V_COLS = [f"V_{i}" for i in range(1, 11)]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
RANDOM_SEED = 42


def safe_read(path):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def ensure_binary_y(y_series):
    if pd.api.types.is_numeric_dtype(y_series):
        vals = sorted(pd.Series(y_series).dropna().unique().tolist())
        if set(vals) <= {0, 1}:
            return y_series.astype(np.int64).values
        raise ValueError(f"y is numeric but not binary 0/1. Found: {vals}")

    y_str = y_series.astype(str).str.strip().str.lower()
    mapping = {
        "fail": 0, "failed": 0, "0": 0,
        "pass": 1, "passed": 1, "1": 1
    }
    if not y_str.isin(mapping.keys()).all():
        bad = sorted(y_str[~y_str.isin(mapping.keys())].unique().tolist())
        raise ValueError(f"Unsupported y labels: {bad}")
    return y_str.map(mapping).astype(np.int64).values


def split_indices(n, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    train_end = int(TRAIN_RATIO * n)
    val_end = int((TRAIN_RATIO + VAL_RATIO) * n)

    return {
        "train": idx[:train_end],
        "val": idx[train_end:val_end],
        "test": idx[val_end:]
    }


def build_onehot_from_original(df_original, df_masked):
    for c in LOW_CARD_CAT_COLS:
        if c not in df_original.columns or c not in df_masked.columns:
            raise ValueError(f"Missing required categorical column: {c}")

    df_original_cat = pd.get_dummies(
        df_original[LOW_CARD_CAT_COLS],
        prefix=LOW_CARD_CAT_COLS,
        dtype=np.float32
    )

    df_masked_cat = pd.get_dummies(
        df_masked[LOW_CARD_CAT_COLS],
        prefix=LOW_CARD_CAT_COLS,
        dtype=np.float32
    )

    # force masked one-hot columns to match original one-hot columns exactly
    df_masked_cat = df_masked_cat.reindex(columns=df_original_cat.columns, fill_value=0.0)

    return df_original_cat, df_masked_cat


def main():
    if not os.path.exists(ORIGINAL_CSV):
        raise FileNotFoundError(f"Original complete CSV not found: {ORIGINAL_CSV}")

    df_original = safe_read(ORIGINAL_CSV)

    if Y_COL not in df_original.columns:
        raise ValueError("Original dataset missing y column")

    df_original = df_original.dropna(subset=[Y_COL]).reset_index(drop=True)

    datasets = sorted([
        d for d in os.listdir(BASE_PATH)
        if d.startswith("dataset_mcar_") and os.path.isdir(os.path.join(BASE_PATH, d))
    ])

    print(f"Found {len(datasets)} MCAR datasets")

    for dataset in datasets:
        dataset_path = os.path.join(BASE_PATH, dataset)
        csv_file = os.path.join(dataset_path, f"{dataset}.csv")

        if not os.path.exists(csv_file):
            print(f"Skipping {dataset}: missing CSV")
            continue

        print(f"\nProcessing {dataset}")
        df_masked = safe_read(csv_file)

        if Y_COL not in df_masked.columns:
            raise ValueError(f"{dataset} missing y column")

        df_masked = df_masked.dropna(subset=[Y_COL]).reset_index(drop=True)

        if len(df_masked) != len(df_original):
            raise ValueError(
                f"Row count mismatch for {dataset}: "
                f"masked={len(df_masked)} vs original={len(df_original)}"
            )

        # same row order required
        original_check_cols = [c for c in ["code_module", "code_presentation", "studied_credits", "num_of_prev_attempts"] if c in df_original.columns and c in df_masked.columns]
        for c in original_check_cols:
            if not df_original[c].equals(df_masked[c]):
                raise ValueError(
                    f"Row/order mismatch detected in column '{c}' for {dataset}. "
                    f"Original and masked files must have identical row order."
                )

        # one-hot from original, align masked to same columns
        df_original_cat, df_masked_cat = build_onehot_from_original(df_original, df_masked)

        all_cols = df_original.columns.tolist()
        num_cols = [c for c in all_cols if c not in LOW_CARD_CAT_COLS + [Y_COL]]

        missing_v = [c for c in V_COLS if c not in num_cols]
        if missing_v:
            raise ValueError(f"Missing V columns: {missing_v}")

        df_original_num = df_original[num_cols].copy()
        df_masked_num = df_masked[num_cols].copy()

        # mask only on V columns from masked csv
        mask_v = (~df_masked_num[V_COLS].isna()).astype(np.int64)

        # model input: masked V -> 0
        df_masked_num[V_COLS] = df_masked_num[V_COLS].fillna(0.0)

        # no NaNs allowed outside V columns
        non_v_cols = [c for c in num_cols if c not in V_COLS]
        if df_masked_num[non_v_cols].isna().any().any():
            bad_cols = df_masked_num[non_v_cols].columns[df_masked_num[non_v_cols].isna().any()].tolist()
            raise ValueError(
                f"{dataset}: found NaNs outside V columns in {bad_cols}"
            )

        # build X_num
        X_main_masked = pd.concat([df_masked_num, df_masked_cat], axis=1)
        X_main_gt = pd.concat([df_original_num, df_original_cat], axis=1)

        # append mask columns at the end
        mask_col_names = [f"M_{c}" for c in V_COLS]
        mask_df = mask_v.copy()
        mask_df.columns = mask_col_names

        X_num_df = pd.concat([X_main_masked, mask_df], axis=1).astype(np.float32)
        X_num_gt_df = pd.concat([X_main_gt, mask_df], axis=1).astype(np.float32)

        final_cols = X_num_df.columns.tolist()
        v_idx = np.array([final_cols.index(c) for c in V_COLS], dtype=np.int64)

        y = ensure_binary_y(df_masked[Y_COL])

        split_idx = split_indices(len(X_num_df), seed=RANDOM_SEED)

        for split, idx in split_idx.items():
            X_num = X_num_df.iloc[idx].values.astype(np.float32)
            X_num_gt = X_num_gt_df.iloc[idx].values.astype(np.float32)
            mask_v_split = mask_v.iloc[idx].values.astype(np.int64)
            y_split = y[idx].astype(np.int64)

            np.save(os.path.join(dataset_path, f"X_num_{split}.npy"), X_num)
            np.save(os.path.join(dataset_path, f"X_num_{split}_gt.npy"), X_num_gt)
            np.save(os.path.join(dataset_path, f"mask_v_{split}.npy"), mask_v_split)
            np.save(os.path.join(dataset_path, f"y_{split}.npy"), y_split)

            print(
                f"{dataset}-{split}: "
                f"X_num={X_num.shape}, "
                f"X_num_gt={X_num_gt.shape}, "
                f"mask_v={mask_v_split.shape}, "
                f"y={y_split.shape}"
            )

        np.save(os.path.join(dataset_path, "v_idx.npy"), v_idx)

        info = {
            "name": dataset,
            "id": dataset,
            "task_type": "binclass",
            "n_classes": 2,
            "n_num_features": int(X_num_df.shape[1]),
            "n_cat_features": 0,
            "train_size": int(len(split_idx["train"])),
            "val_size": int(len(split_idx["val"])),
            "test_size": int(len(split_idx["test"])),
            "v_cols": V_COLS,
            "mask_cols": mask_col_names,
            "num_col_names": final_cols
        }

        with open(os.path.join(dataset_path, "info.json"), "w") as f:
            json.dump(info, f, indent=4)

        print(f"Saved info.json and v_idx.npy for {dataset}")
        print(f"num_numerical_features = {X_num_df.shape[1]}")

    print("\nDONE")


if __name__ == "__main__":
    main()
