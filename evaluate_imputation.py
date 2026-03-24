import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import entropy, gaussian_kde
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse

# -----------------------------
# CHANGE THESE PATHS
# -----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
args = parser.parse_args()

DATASET = args.dataset
DATASET_PATH = f"data/{DATASET}"
IMPUTED_PATH = f"exp/{DATASET}/tabddpm_impute/imputed"
OUT_DIR = f"exp/{DATASET}/tabddpm_impute/eval_imputation"

os.makedirs(OUT_DIR, exist_ok=True)

SPLITS = ["train", "val", "test"]


# -----------------------------
# METRICS
# -----------------------------
def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def safe_kl(p, q, bins=50):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p = p[np.isfinite(p)]
    q = q[np.isfinite(q)]

    if len(p) == 0 or len(q) == 0:
        return None

    lo = min(p.min(), q.min())
    hi = max(p.max(), q.max())

    if lo == hi:
        return 0.0

    p_hist, _ = np.histogram(p, bins=bins, range=(lo, hi), density=True)
    q_hist, _ = np.histogram(q, bins=bins, range=(lo, hi), density=True)

    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    return float(entropy(p_hist, q_hist))


# -----------------------------
# SAFE KDE PLOT
# -----------------------------
def kde_plot(true_vals, pred_vals, title, save_path):
    true_vals = np.asarray(true_vals, dtype=float)
    pred_vals = np.asarray(pred_vals, dtype=float)

    true_vals = true_vals[np.isfinite(true_vals)]
    pred_vals = pred_vals[np.isfinite(pred_vals)]

    if len(true_vals) < 2 or len(pred_vals) < 2:
        print(f"Skipping KDE for {title}: not enough points")
        return

    # if all values are same, add tiny jitter
    if np.std(true_vals) == 0:
        true_vals = true_vals + np.random.normal(0, 1e-6, size=len(true_vals))
    if np.std(pred_vals) == 0:
        pred_vals = pred_vals + np.random.normal(0, 1e-6, size=len(pred_vals))

    lo = min(true_vals.min(), pred_vals.min())
    hi = max(true_vals.max(), pred_vals.max())

    if lo == hi:
        print(f"Skipping KDE for {title}: constant values")
        return

    xs = np.linspace(lo, hi, 300)

    try:
        kde_true = gaussian_kde(true_vals)
        kde_pred = gaussian_kde(pred_vals)

        plt.figure(figsize=(7, 5))
        plt.plot(xs, kde_true(xs), label="Ground Truth")
        plt.plot(xs, kde_pred(xs), label="Imputed")
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Skipping KDE for {title}: {e}")


# -----------------------------
# MAIN
# -----------------------------
def main():
    v_idx = np.load(os.path.join(DATASET_PATH, "v_idx.npy"))

    all_results = {}

    for split in SPLITS:
        print(f"\nEvaluating split: {split}")

        gt = np.load(os.path.join(DATASET_PATH, f"X_num_{split}_gt.npy"))
        mask_v = np.load(os.path.join(DATASET_PATH, f"mask_v_{split}.npy")).astype(bool)
        imputed = np.load(os.path.join(IMPUTED_PATH, f"X_num_{split}_full.npy"))

        split_results = {}

        all_true_masked = []
        all_pred_masked = []

        for j, col_idx in enumerate(v_idx):
            var_name = f"V_{j+1}"
            miss = ~mask_v[:, j]

            true_vals = gt[miss, col_idx]
            pred_vals = imputed[miss, col_idx]

            true_vals = true_vals[np.isfinite(true_vals)]
            pred_vals = pred_vals[np.isfinite(pred_vals)]

            n_masked = min(len(true_vals), len(pred_vals))

            if n_masked == 0:
                split_results[var_name] = {
                    "rmse": None,
                    "mae": None,
                    "kl": None,
                    "n_masked": 0
                }
                continue

            true_vals = true_vals[:n_masked]
            pred_vals = pred_vals[:n_masked]

            this_rmse = rmse(true_vals, pred_vals)
            this_mae = float(mean_absolute_error(true_vals, pred_vals))
            this_kl = safe_kl(true_vals, pred_vals)

            split_results[var_name] = {
                "rmse": this_rmse,
                "mae": this_mae,
                "kl": this_kl,
                "n_masked": int(n_masked)
            }

            all_true_masked.append(true_vals)
            all_pred_masked.append(pred_vals)

            kde_plot(
                true_vals,
                pred_vals,
                title=f"{split} - {var_name}",
                save_path=os.path.join(OUT_DIR, f"kde_{split}_{var_name}.png")
            )

        if len(all_true_masked) == 0:
            split_results["overall"] = {
                "rmse": None,
                "mae": None,
                "kl": None,
                "n_masked": 0
            }
        else:
            all_true_masked = np.concatenate(all_true_masked)
            all_pred_masked = np.concatenate(all_pred_masked)

            split_results["overall"] = {
                "rmse": rmse(all_true_masked, all_pred_masked),
                "mae": float(mean_absolute_error(all_true_masked, all_pred_masked)),
                "kl": safe_kl(all_true_masked, all_pred_masked),
                "n_masked": int(len(all_true_masked))
            }

            kde_plot(
                all_true_masked,
                all_pred_masked,
                title=f"{split} - Overall Masked V Values",
                save_path=os.path.join(OUT_DIR, f"kde_{split}_overall.png")
            )

        all_results[split] = split_results

    out_file = os.path.join(OUT_DIR, "imputation_metrics.json")
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=4)

    print("\nSaved imputation evaluation to:", out_file)
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
