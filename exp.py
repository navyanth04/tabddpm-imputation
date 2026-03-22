import os
import numpy as np

base_data = "data"
base_exp = "exp"

datasets = [d for d in os.listdir(base_data) if d.startswith("dataset_mcar_")]

for d in datasets:
    data_path = os.path.join(base_data, d)
    exp_path = os.path.join(base_exp, d)

    os.makedirs(exp_path, exist_ok=True)

    # read numeric metadata only
    X_num = os.path.join(data_path, "X_num_train.npy")

    if not os.path.exists(X_num):
        print(f"Skipping {d} (missing X_num_train.npy)")
        continue

    Xn = np.load(X_num)
    num_features = Xn.shape[1]

    config = f"""parent_dir = "exp/{d}/tabddpm_impute"
real_data_path = "data/{d}/"
num_numerical_features = {num_features}
model_type = "mlp"
seed = 0
device = "cuda:0"

[model_params]
num_classes = 2
is_y_cond = false

[model_params.rtdl_params]
d_layers = [
    1024,
    1024,
    1024,
    512
]
dropout = 0.1

[diffusion_params]
num_timesteps = 1000
gaussian_loss_type = "mse"
scheduler = "cosine"

[train.main]
steps = 40000
lr = 0.0005
weight_decay = 1e-05
batch_size = 128

[train.T]
seed = 0
normalization = "quantile"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"

[sample]
num_samples = 50000
batch_size = 10000
seed = 0

[eval.type]
eval_model = "catboost"
eval_type = "synthetic"

[eval.T]
seed = 0
normalization = "__none__"
num_nan_policy = "__none__"
cat_nan_policy = "__none__"
cat_min_frequency = "__none__"
cat_encoding = "__none__"
y_policy = "default"
"""

    config_path = os.path.join(exp_path, "config.toml")

    with open(config_path, "w") as f:
        f.write(config)

    print(f"Created config.toml → {config_path}")

print("DONE")
