#!/bin/bash
#SBATCH --job-name=hydrognn-swmm
#SBATCH --array=0-7
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%a.out
# HydroGNN on Bellinge (tasks 0-3) and Tuindorp (4-7) at coverage {1,5,10,20}%.
source scripts/_common.sh

NETS=(bellinge tuindorp)
RATES=(1 5 10 20)
NET=${NETS[$((SLURM_ARRAY_TASK_ID / 4))]}
R=${RATES[$((SLURM_ARRAY_TASK_ID % 4))]}
if [ "$NET" = bellinge ]; then DATA=$BELLINGE_DATA; TOPO=$BELLINGE_TOPO
else                           DATA=$TUINDORP_DATA; TOPO=$TUINDORP_TOPO; fi
OUT=outputs/hydrognn/${NET}_${R}pct
mkdir -p "$OUT"

python hydrognn/train.py \
  --data_file "$DATA" \
  --sensor_file "$SENSORS/sensors_${NET}_kmeans_${R}pct.txt" \
  --topology_csv "$TOPO" \
  --output_dir "$OUT" \
  "${HYDROGNN_ARGS[@]}" "${AUX_ARGS[@]}"
