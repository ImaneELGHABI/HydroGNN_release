#!/bin/bash
#SBATCH --job-name=baselines
#SBATCH --array=0-34
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%a.out
# Homogeneous GNN baselines at 1% coverage (Table 3).
# 5 architectures x {Heusden folds 0-4, Bellinge, Tuindorp}.
source scripts/_common.sh

ARCHS=(gcn gat sage gin gatres)
ARCH=${ARCHS[$((SLURM_ARRAY_TASK_ID % 5))]}
J=$((SLURM_ARRAY_TASK_ID / 5))
case $J in
  [0-4]) DATA="${HEUSDEN_FOLD}${J}.pt"; NET=heusden; OUT=outputs/baselines/heusden_fold${J}/$ARCH ;;
  5)     DATA=$BELLINGE_DATA;           NET=bellinge; OUT=outputs/baselines/bellinge/$ARCH ;;
  6)     DATA=$TUINDORP_DATA;           NET=tuindorp; OUT=outputs/baselines/tuindorp/$ARCH ;;
esac
mkdir -p "$OUT"

python baselines/train_homogeneous.py \
  --data_file "$DATA" \
  --sensor_file "$SENSORS/sensors_${NET}_kmeans_1pct.txt" \
  --arch "$ARCH" --output_dir "$OUT" \
  --hidden_dim 128 --num_layers 4 --heads 4 --dropout 0.1 \
  --epochs 200 --batch_size 16 --lr 1e-3 --weight_decay 1e-5 \
  --patience 40 --seed 42
