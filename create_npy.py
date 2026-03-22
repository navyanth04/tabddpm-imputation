import pandas as pd
import numpy as np
import os

data_path = "data/studentbin5"

# Function to convert CSV → X.npy (features) + y.npy (target)
def csv_to_npy(split):
    df = pd.read_csv(os.path.join(data_path, f"{split}.csv"))
    y = df["y"].values  # target column
    X = df.drop(columns=["y"]).values  # all other columns as features

    np.save(os.path.join(data_path, f"X_{split}.npy"), X)
    np.save(os.path.join(data_path, f"y_{split}.npy"), y)
    print(f"{split}: X_{split}.npy and y_{split}.npy saved!")

# Convert all splits
for split in ["train", "val", "test"]:
    csv_to_npy(split)
