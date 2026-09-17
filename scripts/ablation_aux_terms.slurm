#!/bin/bash
#SBATCH --job-name=ablation-aux
#SBATCH --array=0-14
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%a.out
# Auxiliary-term variants on Heusden at 1%, folds 0-4 (Table S3).
# The reported configuration is outputs/hydrognn/heusden_1pct_fold*.
#   no_aux     both terms off
#   no_warmup  terms at full weight from epoch 1
#   learned    weights learned by homoscedastic uncertainty
source scripts/_common.sh

VARIANTS=(no_aux no_warmup learned)
V=${VARIANTS[$((SLURM_ARRAY_TASK_ID / 5))]}
K=$((SLURM_ARRAY_TASK_ID % 5))
case $V in
  no_aux)    EXTRA=(--lambda_grad 0.0 --lambda_mass 0.0 --min_epochs 30) ;;
  no_warmup) EXTRA=(--corrected_mass --lambda_grad 0.1 --lambda_mass 0.05) ;;
  learned)   EXTRA=(--corrected_mass --lambda_grad 0.1 --lambda_mass 0.05 --learnable_loss_weights) ;;
esac
OUT=outputs/ablations/${V}_fold${K}
mkdir -p "$OUT"

python hydrognn/train.py \
  --data_file "${HEUSDEN_FOLD}${K}.pt" \
  --sensor_file "$SENSORS/sensors_heusden_kmeans_1pct.txt" \
  --topology_csv "$HEUSDEN_TOPO" --datum_patch_units \
  --output_dir "$OUT" \
  "${HYDROGNN_ARGS[@]}" "${EXTRA[@]}"
