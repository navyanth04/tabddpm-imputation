import os
import argparse
import numpy as np
import torch
import zero
import lib

from tab_ddpm import GaussianMultinomialDiffusion
from utils_train import get_model, make_dataset_impute


@torch.no_grad()
def impute(config_path: str, split: str):
    raw_config = lib.load_config(config_path)
    device = torch.device(raw_config["device"])
    zero.improve_reproducibility(raw_config["seed"])

    data_path = os.path.normpath(raw_config["real_data_path"])
    parent_dir = os.path.normpath(raw_config["parent_dir"])

    T = lib.Transformations(**raw_config["train"]["T"])

    # Fit transforms on the same masked trial dataset used for training
    D = make_dataset_impute(
        data_path,
        T,
        change_val=False
    )

    # numeric-only path
    if D.X_cat is not None:
        raise RuntimeError(
            "This impute.py expects numeric-only X_num. "
            "One-hot encode code_module and code_presentation first, "
            "and do not use X_cat for this path."
        )

    K = np.array([0])
    num_numerical_features = D.X_num["train"].shape[1]
    d_in = num_numerical_features
    raw_config["model_params"]["d_in"] = int(d_in)

    model = get_model(
        raw_config["model_type"],
        raw_config["model_params"],
        num_numerical_features,
        category_sizes=[]
    )

    model_path = os.path.join(parent_dir, "model_ema.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(parent_dir, "model.pt")

    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    diffusion = GaussianMultinomialDiffusion(
        num_classes=K,
        num_numerical_features=num_numerical_features,
        denoise_fn=model,
        num_timesteps=raw_config["diffusion_params"]["num_timesteps"],
        gaussian_loss_type=raw_config["diffusion_params"]["gaussian_loss_type"],
        scheduler=raw_config["diffusion_params"]["scheduler"],
        device=device,
    )

    diffusion.to(device)
    diffusion.eval()

    X_num_masked = np.load(os.path.join(data_path, f"X_num_{split}.npy"))
    X_num_gt = np.load(os.path.join(data_path, f"X_num_{split}_gt.npy"))
    mask_v = np.load(os.path.join(data_path, f"mask_v_{split}.npy")).astype(bool)
    v_idx = np.load(os.path.join(data_path, "v_idx.npy"))

    if mask_v.shape[1] != len(v_idx):
        raise RuntimeError("mask_v and v_idx do not match")

    # Full observed mask: everything observed except missing V columns
    obs_mask_full = np.ones_like(X_num_masked, dtype=bool)
    for j, col in enumerate(v_idx):
        obs_mask_full[:, col] = mask_v[:, j]

    # Transform using same transform as training
    X_num_obs_t = D.num_transform.transform(X_num_masked)
    X_num_obs_t = torch.tensor(X_num_obs_t, dtype=torch.float32, device=device)
    obs_mask_t = torch.tensor(obs_mask_full, dtype=torch.bool, device=device)

    # Start reverse process from noise
    z = torch.randn_like(X_num_obs_t)
    fixed_noise = torch.randn_like(X_num_obs_t)

    for i in reversed(range(diffusion.num_timesteps)):
        print(f"Impute timestep {i:4d}", end="\r")
        t = torch.full((X_num_obs_t.shape[0],), i, device=device, dtype=torch.long)

        # observed cells clamped to q(x_t | x_0)
        x_known_t = diffusion.gaussian_q_sample(X_num_obs_t, t, noise=fixed_noise)
        z[obs_mask_t] = x_known_t[obs_mask_t]

        model_out = diffusion._denoise_fn(z.float(), t, y=None)
        z_next = diffusion.gaussian_p_sample(
            model_out,
            z,
            t,
            clip_denoised=False
        )["sample"].float()

        # only update missing cells
        z[~obs_mask_t] = z_next[~obs_mask_t]
        # keep observed cells fixed
        z[obs_mask_t] = x_known_t[obs_mask_t]

    print()

    X_num_imputed = D.num_transform.inverse_transform(z.cpu().numpy())

    # Final completed matrix
    X_num_full = X_num_gt.copy()

    # Replace only missing V entries
    for j, col in enumerate(v_idx):
        miss = ~mask_v[:, j]
        X_num_full[miss, col] = X_num_imputed[miss, col]

    # Last 10 columns are M_V_1..M_V_10 mask features
    # Keep a full copy and also a no-mask copy for fair evaluation / downstream models
    out_dir = os.path.join(parent_dir, "imputed")
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, f"X_num_{split}_full.npy"), X_num_full)

    # drop the 10 mask feature columns at the end
    X_num_nomask = X_num_full[:, :-10]
    np.save(os.path.join(out_dir, f"X_num_{split}.npy"), X_num_nomask)

    # if y exists, copy it
    y_path = os.path.join(data_path, f"y_{split}.npy")
    if os.path.exists(y_path):
        y = np.load(y_path)
        np.save(os.path.join(out_dir, f"y_{split}.npy"), y)

    info_src = os.path.join(data_path, "info.json")
    info_dst = os.path.join(out_dir, "info.json")
    if os.path.exists(info_src):
        import shutil
        shutil.copyfile(info_src, info_dst)

    print(f"Saved imputed split to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    args = parser.parse_args()

    impute(args.config, args.split)
