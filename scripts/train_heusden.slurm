#!/bin/bash
#SBATCH --job-name=hydrognn-heusden
#SBATCH --array=0-19
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%a.out
# HydroGNN on Heusden: coverage {1,5,10,20}% x folds 0-4.
# 1% = Tables 2-3; all coverages = Table 4 and Table S2.
source scripts/_common.sh

RATES=(1 5 10 20)
R=${RATES[$((SLURM_ARRAY_TASK_ID / 5))]}
K=$((SLURM_ARRAY_TASK_ID % 5))
OUT=outputs/hydrognn/heusden_${R}pct_fold${K}
mkdir -p "$OUT"

python hydrognn/train.py \
  --data_file "${HEUSDEN_FOLD}${K}.pt" \
  --sensor_file "$SENSORS/sensors_heusden_kmeans_${R}pct.txt" \
  --topology_csv "$HEUSDEN_TOPO" --datum_patch_units \
  --output_dir "$OUT" \
  "${HYDROGNN_ARGS[@]}" "${AUX_ARGS[@]}"
