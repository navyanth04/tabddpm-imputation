import os
import json
import numpy as np
import pandas as pd

BASE_PATH = "data"
Y_COL = "y"

# these will be one-hot encoded
LOW_CARD_CAT_COLS = ["code_module", "code_presentation"]

# V columns are the only ones with MCAR missingness
V_COLS = [f"V_{i}" for i in range(1, 11)]

# columns that should never go into X
DROP_IF_PRESENT = []

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def safe_read(path):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def ensure_binary_y(y_series):
    # supports numeric 0/1 or string labels like Pass/Fail
    if pd.api.types.is_numeric_dtype(y_series):
        vals = sorted(pd.Series(y_series).dropna().unique().tolist())
        if set(vals) <= {0, 1}:
            return y_series.astype(np.int64).values
        raise ValueError(f"y is numeric but not binary 0/1. Found: {vals}")

    y_str = y_series.astype(str).str.strip().str.lower()
    mapping = {
        "fail": 0,
        "failed": 0,
        "0": 0,
        "pass": 1,
        "passed": 1,
        "1": 1,
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


def build_numeric_table(df):
    # keep y separate
    if Y_COL not in df.columns:
        raise ValueError("Missing y column")

    # only rows with y present
    df = df.dropna(subset=[Y_COL]).copy()

    # one-hot encode code_module and code_presentation
    for c in LOW_CARD_CAT_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required categorical column: {c}")

    df_cat = pd.get_dummies(df[LOW_CARD_CAT_COLS], prefix=LOW_CARD_CAT_COLS, dtype=np.float32)

    # numeric columns = everything except low-card cats and y
    all_cols = df.columns.tolist()
    num_cols = [c for c in all_cols if c not in LOW_CARD_CAT_COLS + [Y_COL] + DROP_IF_PRESENT]

    # make sure V columns exist
    missing_v = [c for c in V_COLS if c not in num_cols]
    if missing_v:
        raise ValueError(f"Missing V columns: {missing_v}")

    df_num = df[num_cols].copy()

    # ground truth before replacing missing V's
    df_num_gt = df_num.copy()

    # create mask only for V columns: 1 observed, 0 missing
    mask_v = (~df_num[V_COLS].isna()).astype(np.int64)

    # replace missing V values with 0 in model input
    df_num[V_COLS] = df_num[V_COLS].fillna(0.0)

    # for non-V numeric columns, do not allow NaNs in this simplified path
    non_v_cols = [c for c in df_num.columns if c not in V_COLS]
    if df_num[non_v_cols].isna().any().any():
        bad_cols = df_num[non_v_cols].columns[df_num[non_v_cols].isna().any()].tolist()
        raise ValueError(
            f"Found NaNs outside V columns in: {bad_cols}. "
            f"Clean these before conversion."
        )

    # append one-hot cat columns
    X_main = pd.concat([df_num, df_cat], axis=1)

    # append mask columns at the end
    mask_col_names = [f"M_{c}" for c in V_COLS]
    mask_df = mask_v.copy()
    mask_df.columns = mask_col_names

    X_with_mask = pd.concat([X_main, mask_df], axis=1)

    # y
    y = ensure_binary_y(df[Y_COL])

    # indices of V columns inside X_num
    final_cols = X_with_mask.columns.tolist()
    v_idx = np.array([final_cols.index(c) for c in V_COLS], dtype=np.int64)

    return {
        "X_num": X_with_mask.astype(np.float32),
        "X_num_gt": pd.concat([df_num_gt, df_cat, mask_df], axis=1).astype(np.float32),
        "mask_v": mask_v.astype(np.int64),
        "y": y,
        "final_cols": final_cols,
        "v_idx": v_idx,
        "n_classes": 2,
    }


def save_split_arrays(dataset_path, built, split_name, idx):
    X_num = built["X_num"].iloc[idx].values.astype(np.float32)
    X_num_gt = built["X_num_gt"].iloc[idx].values.astype(np.float32)
    mask_v = built["mask_v"].iloc[idx].values.astype(np.int64)
    y = built["y"][idx].astype(np.int64)

    np.save(os.path.join(dataset_path, f"X_num_{split_name}.npy"), X_num)
    np.save(os.path.join(dataset_path, f"X_num_{split_name}_gt.npy"), X_num_gt)
    np.save(os.path.join(dataset_path, f"mask_v_{split_name}.npy"), mask_v)
    np.save(os.path.join(dataset_path, f"y_{split_name}.npy"), y)

    print(
        f"{split_name}: X_num={X_num.shape}, "
        f"X_num_gt={X_num_gt.shape}, mask_v={mask_v.shape}, y={y.shape}"
    )


def write_info_json(dataset_path, built):
    info = {
        "task_type": "binclass",
        "n_classes": int(built["n_classes"]),
        "name": os.path.basename(dataset_path),
        "num_col_names": built["final_cols"],
        "target_col": Y_COL,
        "v_cols": V_COLS,
    }
    with open(os.path.join(dataset_path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)


def main():
    datasets = [
        d for d in os.listdir(BASE_PATH)
        if d.startswith("dataset_mcar_") and os.path.isdir(os.path.join(BASE_PATH, d))
    ]

    print(f"Found {len(datasets)} MCAR datasets")

    for dataset in sorted(datasets):
        dataset_path = os.path.join(BASE_PATH, dataset)
        csv_file = os.path.join(dataset_path, f"{dataset}.csv")

        if not os.path.exists(csv_file):
            print(f"Skipping {dataset}: missing CSV {csv_file}")
            continue

        print(f"\nProcessing {dataset}")
        df = safe_read(csv_file)

        built = build_numeric_table(df)

        # save v_idx once
        np.save(os.path.join(dataset_path, "v_idx.npy"), built["v_idx"])

        # save split arrays
        split_idx = split_indices(len(built["X_num"]), seed=RANDOM_SEED)
        for split_name, idx in split_idx.items():
            save_split_arrays(dataset_path, built, split_name, idx)

        write_info_json(dataset_path, built)

        print(f"Saved info.json and v_idx.npy for {dataset}")
        print(f"num_numerical_features = {built['X_num'].shape[1]}")

    print("\nDONE")


if __name__ == "__main__":
    main()
