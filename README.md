# HydroGNN

Code for *Graph Representation Learning for Hydraulic State Reconstruction in
Urban Drainage under Limited Observability* (under review).

HydroGNN is a heterogeneous graph neural network that estimates the water level
at every manhole of a sewer network from a small number of instrumented
manholes (1% in the paper). Pipes, weirs and pumps are represented as nodes
(lines-as-nodes), and messages are passed with type-pair specific attention
(TypePairMP).

## Layout

```
hydrognn/        model, training loop, loss terms, lines-as-nodes transform
baselines/       homogeneous GNN baselines (GCN, GAT, GraphSAGE, GIN, GATRes)
preprocessing/   dataset construction for Bellinge, Tuindorp and Heusden; k-means sensor placement
scripts/         SLURM scripts for every training run in the paper
analysis/        tables and numbers quoted in the paper
figures/         Figures 1, 5, 6 and 7
data/sensors/    sensor sets (node indices) at 1, 5, 10 and 20% coverage
data/heusden_folds.json   storm-to-fold assignment for Heusden
```

## Installation

Python 3.11 or later and a CUDA build of PyTorch. The paper runs used PyTorch
2.5.1 and PyTorch Geometric 2.7.0.

```
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Run all commands from the repository root with `export PYTHONPATH=$PWD`.

## Data

| Network  | Source | Location |
|----------|--------|----------|
| Bellinge | Nedergaard Pedersen et al. (2021), *Earth Syst. Sci. Data* 13, 4779–4798, https://doi.org/10.5194/essd-13-4779-2021 | `data/raw/bellinge/` (`7_SWMM/`, `2_cleaned_data/`) |
| Tuindorp | Garzón et al. (2024), 4TU.ResearchData, https://doi.org/10.4121/fec1e3de-9586-4a61-b3a1-02382592e52c.v1 | `data/raw/tuindorp/Tuindorp development - 1 min resolution/` |
| Heusden  | Provided by the network operator and not redistributable. Available on reasonable request through the corresponding author. | `data/raw/heusden/` |

### Bellinge

```
python preprocessing/bellinge/generate_event_list.py
python preprocessing/bellinge/build_topology.py
# one SWMM run per row of data/interim/bellinge/event_list.csv
python preprocessing/bellinge/run_event.py --start <start> --end <end> --tag <tag>
python preprocessing/bellinge/build_dataset.py
python preprocessing/bellinge/strip_pump_status.py
```

Output: `data/processed/bellinge_v3_nopumpstatus.pt`

### Tuindorp

```
python preprocessing/tuindorp/build_dataset.py
python preprocessing/tuindorp/attach_inflow.py
```

Output: `data/processed/tuindorp_v2_rain.pt`

### Heusden

```
python preprocessing/heusden/build_event_folds.py
python preprocessing/heusden/build_fold_datasets.py
python preprocessing/heusden/fix_units_si.py
```

Output: `data/processed/heusden_v3/folds/heusden_v3_fold{0..4}.pt`. The storm
assignment per fold is in `data/heusden_folds.json`.

### Sensor sets

`data/sensors/` holds the sets used in the paper. To regenerate them:

```
python preprocessing/kmeans_sensors_swmm.py bellinge 0.01
python preprocessing/kmeans_sensors_swmm.py tuindorp 0.01
python preprocessing/kmeans_sensors_heusden.py 0.01
```

## Training

Settings shared by all runs are in `scripts/_common.sh`. Submit from the
repository root:

```
sbatch scripts/train_heusden.slurm       # 4 coverages x 5 folds
sbatch scripts/train_swmm.slurm          # Bellinge and Tuindorp, 4 coverages
sbatch scripts/train_baselines.slurm     # 5 baselines x (5 Heusden folds, Bellinge, Tuindorp)
sbatch scripts/ablation_aux_terms.slurm  # Table S3
sbatch scripts/ablation_hydraulic.slurm  # weir and pump terms, Section S8.2
```

Each run writes `best_model.pth` and `test_metrics.json` under `outputs/`.
All runs use seed 42. Without SLURM, set `SLURM_ARRAY_TASK_ID` and run the
script with `bash`.

## Reproducing the paper

Run `python -m analysis.predict_test` first. It writes the test predictions
that Table 2, Section 4.5 and the figures read. Results are printed and
written to `results/`.

| Paper | Command |
|-------|---------|
| Table 1 | `python -m analysis.table1_datasets` |
| Table 2, HydroGNN | `python -m analysis.table2_hydrognn` |
| Table 2, climatology and nearest sensor | `python -m analysis.table2_references` |
| Table 3, Sections 4.2 and 4.4 | `python -m analysis.table3_baselines` |
| Table 4, Table S2 | `python -m analysis.table4_coverage` |
| Table 5, Section 5.1 | `python -m analysis.real_sensor.sim_vs_measured`<br>`python -m analysis.real_sensor.model_vs_measured` |
| Table S3, Section S8.2 | `python -m analysis.table_s3_aux_terms` |
| Section 3.1.2 | `python -m analysis.storm_similarity`<br>`python -m analysis.fold_breakdown --runs "outputs/hydrognn/heusden_1pct_fold{k}" --tag heusden_1pct` |
| Section 3.2.1, Section S5 | `python -m analysis.masking_audit` |
| Section 4.5 | `python -m analysis.spatial_error` |
| Figure 1 | `python -m figures.fig1_networks` |
| Figure 5 | `python -m figures.fig5_hydrographs` |
| Figures 6 and 7 | `python -m figures.fig6_7_error_maps` |

Figures 2–4 are diagrams. Table S4 (Section S8.3) was produced with an
earlier version of the code (12 blocks, climatology channels at every
manhole) and is not reproduced by these scripts.

## License

MIT, see `LICENSE`.
# HydroGNN_release
