# Sourced by the SLURM scripts. Run sbatch from the repository root.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

SENSORS=data/sensors
HEUSDEN_FOLD=data/processed/heusden_v3/folds/heusden_v3_fold
HEUSDEN_TOPO="data/raw/heusden/Detailed_heusden_topology/1D rioleringsmodel Heusden!_node.csv"
BELLINGE_DATA=data/processed/bellinge_v3_nopumpstatus.pt
BELLINGE_TOPO=data/interim/bellinge/topology/BellingeSWMM_node.csv
TUINDORP_DATA=data/processed/tuindorp_v2_rain.pt
TUINDORP_TOPO="data/raw/tuindorp/Tuindorp development - 1 min resolution/networks/Tuindorp development - 1 min resolution_node.csv"

# Settings shared by every HydroGNN run in the paper.
HYDROGNN_ARGS=(
  --data_format spatial --datum_mode off
  --global_context --wetness_features --input_skip
  --use_system_type --use_type_weighted_loss
  --pred_loss mae --add_qmax_feature
  --epochs 80 --batch_size 16 --lr 1e-3 --weight_decay 1e-5
  --hidden_dim 128 --num_blocks 4 --heads 4 --dropout 0.1 --patience 25
  --lambda_pred 1.0
  --lambda_flow 0.0 --lambda_qmax 0.0 --lambda_depth 0.0 --lambda_level 0.0
  --seed 42
)
# Auxiliary terms as reported (Section 3.6).
AUX_ARGS=(--corrected_mass --lambda_grad 0.1 --lambda_mass 0.05
          --physics_warmup --physics_warmup_epochs 30)
