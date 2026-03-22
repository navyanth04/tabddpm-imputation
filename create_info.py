import os
import json
import numpy as np

base_path = "data"

datasets = [d for d in os.listdir(base_path) if d.startswith("dataset_mcar_")]

for d in datasets:
    path = os.path.join(base_path, d)

    X_num = np.load(os.path.join(path, "X_num_train.npy"))
    X_cat = np.load(os.path.join(path, "X_cat_train.npy"))

    val = np.load(os.path.join(path, "X_num_val.npy")).shape[0]
    test = np.load(os.path.join(path, "X_num_test.npy")).shape[0]
    train = X_num.shape[0]

    info = {
        "name": d,
        "id": d,
        "task_type": "binclass",
        "n_num_features": X_num.shape[1],
        "n_cat_features": X_cat.shape[1],
        "train_size": train,
        "val_size": val,
        "test_size": test
    }

    with open(os.path.join(path, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    print(f"Created info.json for {d}")

print("DONE")
