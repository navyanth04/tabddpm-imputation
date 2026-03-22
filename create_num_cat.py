import pandas as pd
import numpy as np
import os

data_path = "data/studentbin5"

cat_cols = ["code_module", "code_presentation"]
y_col = "y"

# automatically take all other columns as numerical
all_cols = pd.read_csv(os.path.join(data_path, "train.csv")).columns.tolist()
num_cols = [c for c in all_cols if c not in cat_cols + [y_col]]

for split in ["train", "val", "test"]:
    df = pd.read_csv(os.path.join(data_path, f"{split}.csv"))
    
    X_num = df[num_cols].values
    X_cat = df[cat_cols].values
    y = df[y_col].values
    
    np.save(os.path.join(data_path, f"X_num_{split}.npy"), X_num)
    np.save(os.path.join(data_path, f"X_cat_{split}.npy"), X_cat)
    np.save(os.path.join(data_path, f"y_{split}.npy"), y)
    
    print(f"{split}: X_num {X_num.shape}, X_cat {X_cat.shape}, y {y.shape}")
