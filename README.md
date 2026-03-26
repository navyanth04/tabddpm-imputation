
# TabDDPM Imputation

TabDDPM Imputer is a tabular imputation pipeline for running MCAR missing-value experiments on datasets stored as `dataset_mcar_*_trial*`.

It supports:

- building TabDDPM-ready arrays from CSV files
- creating one config per dataset
- training the model
- imputing missing values for `train`, `val`, and `test`
- evaluating imputation quality
- evaluating downstream prediction performance
- summarizing results across trials

---

## Repository structure

```text
.
├── data/
├── exp/
├── lib/
├── scripts/
├── tab_ddpm/
├── convert_to_tabddpm_impute_from_original.py
├── evaluate_imputation.py
├── evaluate_downstream.py
├── exp.py
├── run_all_datasets.sh
├── summarize_trials.py
└── requirements.txt

```

### Main files

-   `convert_to_tabddpm_impute_from_original.py`  
    Builds the `.npy` files used by training and imputation.
    
-   `exp.py`  
    Generates `config.toml` for each dataset under `exp/`.
    
-   `scripts/pipeline.py`  
    Starts model training.
    
-   `scripts/train.py`  
    Contains the training loop.
    
-   `scripts/impute.py`  
    Loads a trained model and imputes missing values.
    
-   `evaluate_imputation.py`  
    Computes imputation metrics and KDE plots.
    
-   `evaluate_downstream.py`  
    Runs downstream prediction on the imputed data.
    
-   `summarize_trials.py`  
    Aggregates results across all trials and missing rates.
    
-   `run_all_datasets.sh`  
    Runs the full pipeline automatically for all datasets in `exp/dataset_mcar_*_trial*`.
    

----------

## Environment setup

Install Conda first, then run:

```bash
export REPO_DIR=/path/to/tabddpm-imputation
cd $REPO_DIR

conda create -n tddpm python=3.9.7 -y
conda activate tddpm

# Example PyTorch install for CUDA 11.1
pip install torch==1.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

pip install -r requirements.txt
pip install matplotlib xgboost

# Optional but convenient
conda env config vars set PYTHONPATH=${PYTHONPATH}:${REPO_DIR}
conda env config vars set PROJECT_DIR=${REPO_DIR}

conda deactivate
conda activate tddpm

```

If you do not want to set env vars, you can still run all commands with `PYTHONPATH=.`.

----------

## Expected data layout

The pipeline expects:

```text
data/
├── original_complete/
│   └── oula_complete.csv
├── dataset_mcar_10_trial1/
│   └── dataset_mcar_10_trial1.csv
├── dataset_mcar_10_trial2/
│   └── dataset_mcar_10_trial2.csv
...
├── dataset_mcar_50_trial5/
│   └── dataset_mcar_50_trial5.csv

```

After conversion, each `data/dataset_mcar_*_trial*/` folder will also contain generated `.npy` files and `info.json`.

----------

## Full workflow

## Step 1: build TabDDPM-ready arrays

Run this when:

-   the original CSV changed
    
-   the masked CSVs changed
    
-   the split logic changed
    
-   the feature or mask logic changed
    

```bash
PYTHONPATH=. python convert_to_tabddpm_impute_from_original.py

```

This creates files such as:

-   `X_num_train.npy`, `X_num_val.npy`, `X_num_test.npy`
    
-   `X_num_train_gt.npy`, `X_num_val_gt.npy`, `X_num_test_gt.npy`
    
-   `mask_v_train.npy`, `mask_v_val.npy`, `mask_v_test.npy`
    
-   `y_train.npy`, `y_val.npy`, `y_test.npy`
    
-   `v_idx.npy`
    
-   `info.json`
    

----------

## Step 2: generate experiment configs

```bash
python exp.py

```

This creates one config per dataset, for example:

```text
exp/dataset_mcar_10_trial1/config.toml
exp/dataset_mcar_10_trial2/config.toml
...
exp/dataset_mcar_50_trial5/config.toml

```

----------

## Step 3: run one dataset manually

Example: `dataset_mcar_10_trial1`

### Train

```bash
PYTHONPATH=. python scripts/pipeline.py --config exp/dataset_mcar_10_trial1/config.toml --train

```

### Impute train split

```bash
PYTHONPATH=. python scripts/impute.py --config exp/dataset_mcar_10_trial1/config.toml --split train

```

### Impute validation split

```bash
PYTHONPATH=. python scripts/impute.py --config exp/dataset_mcar_10_trial1/config.toml --split val

```

### Impute test split

```bash
PYTHONPATH=. python scripts/impute.py --config exp/dataset_mcar_10_trial1/config.toml --split test

```

### Evaluate imputation

```bash
PYTHONPATH=. python evaluate_imputation.py --dataset dataset_mcar_10_trial1

```

### Evaluate downstream prediction

```bash
PYTHONPATH=. python evaluate_downstream.py --dataset dataset_mcar_10_trial1

```

----------

## Step 4: run all datasets automatically

You can use the provided shell script:

```bash
bash run_all_datasets.sh

```

This runs, in order, for every dataset folder under `exp/dataset_mcar_*_trial*`:

1.  train the model
    
2.  impute `train`
    
3.  impute `val`
    
4.  impute `test`
    
5.  evaluate imputation
    
6.  evaluate downstream prediction
    

It also stores:

-   per-dataset logs
    
-   a summary file with `OK` / `FAILED`
    
-   experiment snapshots
    

----------

## Step 5: summarize all trial results

After all runs finish:

```bash
PYTHONPATH=. python summarize_trials.py

```

This creates:

```text
summary_results/
├── all_trials_metrics.csv
├── avg_by_missing_rate.csv
└── std_by_missing_rate.csv

```

----------

## What `run_all_datasets.sh` does

The script:

-   creates a run folder under `$HOME/imputation_results/tabddpm/`
    
-   creates a log directory
    
-   creates an experiment snapshot directory
    
-   loops over all folders matching `exp/dataset_mcar_*_trial*`
    
-   uses each folder’s `config.toml`
    
-   runs training, imputation, and both evaluations
    
-   saves a separate log for each dataset
    
-   writes `summary.txt` and `failed.txt`
    
-   copies a snapshot of each finished experiment folder
    

If you want full control, you can skip the shell script and run the commands manually as shown above.

----------

## Output files

For each dataset, the main experiment output folder is:

```text
exp/dataset_mcar_xx_trialy/tabddpm_impute/

```

### Training outputs

-   `config.toml`
    
-   `info.json`
    
-   `loss.csv`
    
-   `model.pt`
    
-   `model_ema.pt`
    

### Imputation outputs

Inside:

```text
exp/dataset_mcar_xx_trialy/tabddpm_impute/imputed/

```

You will find:

-   `X_num_train.npy`
    
-   `X_num_val.npy`
    
-   `X_num_test.npy`
    
-   `X_num_train_full.npy`
    
-   `X_num_val_full.npy`
    
-   `X_num_test_full.npy`
    
-   `y_train.npy`
    
-   `y_val.npy`
    
-   `y_test.npy`
    
-   `info.json`
    

### Imputation evaluation outputs

Inside:

```text
exp/dataset_mcar_xx_trialy/tabddpm_impute/eval_imputation/

```

You will find:

-   `imputation_metrics.json`
    
-   KDE plot images such as:
    
    -   `kde_train_V_1.png`
        
    -   `kde_val_V_5.png`
        
    -   `kde_test_overall.png`
        

### Downstream evaluation output

-   `exp/dataset_mcar_xx_trialy/tabddpm_impute/downstream_metrics.json`
    

----------

## Typical command order

If you are running everything manually, the order is:

```bash
PYTHONPATH=. python convert_to_tabddpm_impute_from_original.py
python exp.py

PYTHONPATH=. python scripts/pipeline.py --config exp/dataset_mcar_10_trial1/config.toml --train
PYTHONPATH=. python scripts/impute.py --config exp/dataset_mcar_10_trial1/config.toml --split train
PYTHONPATH=. python scripts/impute.py --config exp/dataset_mcar_10_trial1/config.toml --split val
PYTHONPATH=. python scripts/impute.py --config exp/dataset_mcar_10_trial1/config.toml --split test
PYTHONPATH=. python evaluate_imputation.py --dataset dataset_mcar_10_trial1
PYTHONPATH=. python evaluate_downstream.py --dataset dataset_mcar_10_trial1

PYTHONPATH=. python summarize_trials.py

```

----------

## Troubleshooting

### `ModuleNotFoundError` for `sample`, `eval_catboost`, `eval_mlp`, or `eval_simple`

`pipeline.py` may still import these helper files at startup. Keep these files inside `scripts/` unless you have already cleaned up the imports in `pipeline.py`.

### Training ran before, but no new loss is printed

If `scripts/pipeline.py --train` crashes and `scripts/impute.py` still runs afterward, then imputation may be using an old saved model. Check the training log first.

### Only a warning is printed about `pkg_resources`

This is usually just a warning, not a fatal error.

### Want to monitor a running experiment

Use:

```bash
pgrep -af "pipeline.py|impute.py|evaluate_imputation.py|evaluate_downstream.py"

```

and for logs:

```bash
tail -f /path/to/logfile.log

```

----------

## Minimal quick start

If your CSV files are already in place and you want to test one dataset quickly:

```bash
conda activate tddpm
cd /path/to/tabddpm-imputation

PYTHONPATH=. python convert_to_tabddpm_impute_from_original.py
python exp.py

PYTHONPATH=. python scripts/pipeline.py --config exp/dataset_mcar_10_trial1/config.toml --train
PYTHONPATH=. python scripts/impute.py --config exp/dataset_mcar_10_trial1/config.toml --split train
PYTHONPATH=. python scripts/impute.py --config exp/dataset_mcar_10_trial1/config.toml --split val
PYTHONPATH=. python scripts/impute.py --config exp/dataset_mcar_10_trial1/config.toml --split test
PYTHONPATH=. python evaluate_imputation.py --dataset dataset_mcar_10_trial1
PYTHONPATH=. python evaluate_downstream.py --dataset dataset_mcar_10_trial1

```

If you want to run all datasets instead:

```bash
bash run_all_datasets.sh

```

```
::contentReference[oaicite:1]{index=1}

```
