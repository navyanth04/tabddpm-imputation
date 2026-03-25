#!/bin/bash
set -uo pipefail

RUNROOT="$HOME/imputation_results/tabddpm/run_$(date +%Y%m%d_%H%M%S)"
LOGDIR="$RUNROOT/logs"
SNAPDIR="$RUNROOT/exp_snapshots"

mkdir -p "$LOGDIR" "$SNAPDIR"

for DATASET_DIR in exp/dataset_mcar_*_trial*; do
    [ -d "$DATASET_DIR" ] || continue

    DATASET=$(basename "$DATASET_DIR")
    CONFIG="$DATASET_DIR/config.toml"

    [ -f "$CONFIG" ] || continue

    echo "===================================="
    echo "Running $DATASET"
    echo "Config: $CONFIG"
    echo "===================================="

    if (
        set -e
        PYTHONPATH=. python scripts/pipeline.py --config "$CONFIG" --train
        PYTHONPATH=. python scripts/impute.py --config "$CONFIG" --split train
        PYTHONPATH=. python scripts/impute.py --config "$CONFIG" --split val
        PYTHONPATH=. python scripts/impute.py --config "$CONFIG" --split test
        PYTHONPATH=. python evaluate_imputation.py --dataset "$DATASET"
        PYTHONPATH=. python evaluate_downstream.py --dataset "$DATASET"
    ) 2>&1 | tee "$LOGDIR/${DATASET}.log"; then
        echo "$DATASET OK" | tee -a "$LOGDIR/summary.txt"
    else
        echo "$DATASET FAILED" | tee -a "$LOGDIR/summary.txt"
        echo "$DATASET" >> "$LOGDIR/failed.txt"
    fi

    # save a snapshot of the whole experiment folder after each dataset
    rm -rf "$SNAPDIR/$DATASET"
    cp -r "$DATASET_DIR" "$SNAPDIR/$DATASET"
done

echo "All done."
echo "Logs are in: $LOGDIR"
echo "Snapshots are in: $SNAPDIR"
