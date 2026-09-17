#!/bin/bash
#SBATCH --job-name=ablation-hydraulic
#SBATCH --array=0-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%a.out
# Explicit weir/pump terms (hydrognn/physics_hydraulic.py) on Heusden folds 0
# and 4, compared with the no_aux control (Section S8.2).
source scripts/_common.sh

FOLDS=(0 4 0 4 0 4)
K=${FOLDS[$SLURM_ARRAY_TASK_ID]}
case $((SLURM_ARRAY_TASK_ID / 2)) in
  0) TAG=full;   LA=0.05; LW=0.05; LB=0.05 ;;
  1) TAG=unsup;  LA=0.0;  LW=0.05; LB=0.05 ;;
  2) TAG=strong; LA=0.2;  LW=0.2;  LB=0.2  ;;
esac
OUT=outputs/ablations/hydraulic_${TAG}_fold${K}
mkdir -p "$OUT"

python hydrognn/train.py \
  --data_file "${HEUSDEN_FOLD}${K}.pt" \
  --sensor_file "$SENSORS/sensors_heusden_kmeans_1pct.txt" \
  --topology_csv "$HEUSDEN_TOPO" --datum_patch_units \
  --output_dir "$OUT" \
  --hydraulic_physics --target_frame level \
  --lambda_asset $LA --lambda_weir_dry $LW --lambda_bounds $LB \
  --lambda_grad 0.0 --lambda_mass 0.0 --min_epochs 30 \
  "${HYDROGNN_ARGS[@]}"
