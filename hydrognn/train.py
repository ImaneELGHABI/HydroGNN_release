"""Train HydroGNN.

Data formats (--data_format, auto-detected by default):
  spatial  .pt with {graphs, train_size, val_size, test_size, depth_mean, depth_std, ...}
  legacy   .pt with {train_graphs, val_graphs}, converted with lines_as_nodes()

Masking: a fixed sensor set with --sensor_file (one node index per line),
otherwise a random set drawn each epoch.

The exact settings used in the paper are in scripts/_common.sh. Example:

  python hydrognn/train.py \\
    --data_file data/processed/tuindorp_v2_rain.pt \\
    --sensor_file data/sensors/sensors_tuindorp_kmeans_1pct.txt \\
    --output_dir outputs/hydrognn/tuindorp_1pct  [settings from scripts/_common.sh]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from hydrognn.data_loader import (lines_as_nodes, strip_asset_physics,
                                  asset_endpoint_indices,
                                  ASSET_GROUP_NUM_SUBTYPES)
from hydrognn.datum import enforce_datum, patch_invert_units
from hydrognn.physics_hydraulic import hydraulic_physics_loss
from hydrognn.asset_features import (relativise_asset_thresholds,
                                     chamber_depth_from_topology)

FEET_PER_METRE_TRAIN = 3.280839895013123
from hydrognn.model import HydroGNN, schema_from_graph
from hydrognn.physics_loss import LearnableLossWeights, physics_loss


# ── utilities ────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_random_mask(num_nodes: int, mask_rate: float, rng: torch.Generator) -> torch.Tensor:
    """True = masked (non-sensor). Random sensor placement."""
    num_sensors = max(1, int(round(num_nodes * (1.0 - mask_rate))))
    mask = torch.ones(num_nodes, dtype=torch.bool)
    mask[torch.randperm(num_nodes, generator=rng)[:num_sensors]] = False
    return mask


def build_fixed_mask(num_nodes: int, sensor_indices: List[int]) -> torch.Tensor:
    """True = masked (non-sensor). Fixed sensor positions from file."""
    mask = torch.ones(num_nodes, dtype=torch.bool)
    valid = [i for i in sensor_indices if 0 <= i < num_nodes]
    if valid:
        mask[torch.tensor(valid, dtype=torch.long)] = False
    return mask


def apply_mask(g: HeteroData, mask: torch.Tensor,
               depth_idxs: List[int], sensor_flag_idx: int) -> HeteroData:
    """Zero out all dynamic depth features for non-sensor nodes; set sensor_flag."""
    g = g.clone()
    x = g["manhole"].x.clone()
    for i in depth_idxs:
        if x.size(1) > i:
            x[mask, i] = 0.0
    if x.size(1) > sensor_flag_idx:
        x[mask,  sensor_flag_idx] = 0.0
        x[~mask, sensor_flag_idx] = 1.0
    g["manhole"].x = x
    g["manhole"].non_sensor_mask = mask
    return g


def compute_climatology(graphs: list,
                        sensor_mask: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-node mean/std of the target over the given (train) snapshots, z-norm units.

    `sensor_mask` (True = observed/sensor node) restricts the statistics to nodes that
    are actually instrumented: their values are kept, every other node gets zero.

    The statistics are read from the targets, so without the mask every manhole would
    receive its own training-period mean as an input. With the mask only sensors keep
    their values, which is what a deployment has.

    sensor_mask=None computes them at every node (--leaky_climatology).
    """
    Y = torch.stack([g["manhole"].y[:, 0] if g["manhole"].y.dim() == 2 else g["manhole"].y
                     for g in graphs])
    mu, sd = Y.mean(0), Y.std(0).clamp(min=1e-6)
    if sensor_mask is not None:
        sensor_mask = sensor_mask.to(mu.device)
        zero = torch.zeros((), dtype=mu.dtype, device=mu.device)
        mu = torch.where(sensor_mask, mu, zero)
        sd = torch.where(sensor_mask, sd, zero)
    return mu, sd


def subtract_target_offset(graphs: list, mu: torch.Tensor) -> None:
    """Reparameterise the target as an anomaly about a fixed per-node offset.

    On Heusden the target is water level in metres above datum, so each node's
    invert elevation enters additively. The per-node error bias of a trained model
    correlates with the node's mean level at r = -0.94 and is uncorrelated with
    distance to the nearest sensor, i.e. the model tracks the dynamics but
    mean-reverts on the static component. Predicting y - mu removes that component
    from the learning problem; the level is recovered as y_hat + mu at inference.

    `mu` must be the per-node mean over the TRAIN snapshots only, in the same
    normalised units as y, and must be computed before this call. MAE and RMSE are
    invariant under the shift because it cancels in (pred - target); only
    statistics that need the absolute level (the wet-record criterion) must add it
    back.

    Modifies y in place, and records the offset on each graph so that any
    downstream consumer can reconstruct absolute levels.
    """
    mu = mu.detach().clone()
    for g in graphs:
        y = g["manhole"].y
        if y.dim() == 2:
            y[:, 0] -= mu
        else:
            y -= mu
        g["manhole"].target_offset = mu


def add_sensor_context_features(g: HeteroData, clim_mu: torch.Tensor, clim_sd: torch.Tensor,
                                 cur_idx: int, include_global: bool = False) -> HeteroData:
    """Append per-node context features, and optionally network-wide ones.

    Two per-node climatology channels (training-period mean and std of the target
    at that manhole) are always appended, but they carry values only where
    compute_climatology was allowed to keep them. Called with a sensor mask, it zeroes
    every unmonitored node, so those two columns are non-zero at instrumented manholes
    and zero elsewhere. That restriction is what stops the target mean at a node the
    model must predict from arriving as one of its own input features.

    The 4 network-wide order statistics of the current sensor readings (mean,
    max, 90th percentile, std, broadcast identically to every node) are appended
    only when include_global is True. They are off by default here: broadcast to
    every node they let the readout regress a regional mean rather than infer a
    local level from propagated sensor readings, and because they are invariant
    to a permutation among sensor locations they also blunt the permutation
    control that is supposed to detect exactly that.

    Note the evidence is not one-sided. The shortcut reading was not supported by
    measurement: the trained model tracks the truth at r = 0.975 one hop from a
    sensor and sits about 44 cm away from its own climatology, and an earlier run
    without the context channels collapsed to roughly 26-29 cm, far worse than
    the 18 cm homogeneous baselines. Keep this flag available in both positions
    and compare rather than assuming removal helps.

    Must be called after apply_mask (reads only sensor nodes via non_sensor_mask).
    """
    x = g["manhole"].x
    parts = [x]
    if include_global:
        mask = g["manhole"].non_sensor_mask
        sensed = x[~mask, cur_idx]
        if sensed.numel() > 0:
            stats = torch.stack([sensed.mean(), sensed.max(),
                                 sensed.quantile(0.9), sensed.std(unbiased=False)])
        else:
            stats = x.new_zeros(4)
        parts.append(stats.unsqueeze(0).expand(x.size(0), 4))
    parts += [clim_mu.unsqueeze(1), clim_sd.unsqueeze(1)]
    g["manhole"].x = torch.cat(parts, dim=1)
    return g


def load_sensor_indices(p: Path) -> List[int]:
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(int(line.split()[0]))
    return sorted(out)


def load_system_type_idx(topology_csv: Path, num_manholes: int) -> torch.Tensor:
    """Load per-manhole sewer system type from topology CSV. Returns LongTensor [N]."""
    import pandas as pd
    sys_map = {"combined": 0, "storm": 1, "foul": 2, "other": 3}
    df = pd.read_csv(topology_csv, encoding="latin-1")
    type_col = next((c for c in ("node_type", "type") if c in df.columns), None)
    mh = (df[df[type_col].astype(str).str.lower() == "manhole"].reset_index(drop=True)
          if type_col else df.reset_index(drop=True))
    if "system_type" not in mh.columns:
        return torch.full((num_manholes,), 3, dtype=torch.long)
    raw = mh["system_type"].fillna("other").astype(str).str.strip().str.lower().values
    idx = ([sys_map.get(s, 3) for s in raw] + [3] * num_manholes)[:num_manholes]
    return torch.tensor(idx, dtype=torch.long)


# ── data loading ─────────────────────────────────────────────────────────────

def filter_dry_snapshots(graphs: list, depth_mean: float, depth_std: float,
                          min_mean_depth_m: float) -> list:
    """Drop snapshots where mean physical depth across all manholes is below threshold.

    Dry snapshots (near-zero depth everywhere) add no learning signal and
    are especially abundant in Bellinge/Tuindorp where 30-66% of timesteps
    are nearly dry. Filtering them focuses training on informative flood events.
    """
    if min_mean_depth_m <= 0:
        return graphs
    kept = []
    for g in graphs:
        y = g["manhole"].y[:, 0] if g["manhole"].y.dim() == 2 else g["manhole"].y
        mean_phys = float(y.mean()) * depth_std + depth_mean
        if mean_phys >= min_mean_depth_m:
            kept.append(g)
    return kept


def get_physics_weights(epoch, warm_up_epochs=30, total_epochs=400,
                        max_lambda_grad=0.01, max_lambda_mass=0.01, floor=0.0):
    """Cosine ramp for the physics penalty weights, starting at the first epoch.

    `warm_up_epochs` is the number of epochs taken to REACH full weight, not a
    number of epochs to sit at zero:

        epoch 1                -> floor * max_lambda
        epoch warm_up_epochs   -> max_lambda
        later epochs           -> max_lambda (held)

    `total_epochs` is unused.

    Returns (lambda_grad, lambda_mass) for the given epoch.
    """
    progress = (epoch - 1) / max(1, warm_up_epochs - 1)
    ramp = 0.5 * (1.0 - math.cos(math.pi * min(1.0, max(0.0, progress))))
    scale = floor + (1.0 - floor) * ramp
    return max_lambda_grad * scale, max_lambda_mass * scale


def load_data(data_file: Path, fmt: str, min_mean_depth_m: float = 0.0,
              add_qmax_feature: bool = False,
              no_asset_physics: bool = False,
              add_manhole_shortcut: bool = False,
              normalize_assets: bool = False) -> Tuple:
    """Returns (train_graphs, val_graphs, depth_mean, depth_std,
                node_dims, edge_dims, edge_types, depth_idxs, sensor_flag_idx)."""
    raw = torch.load(data_file, map_location="cpu", weights_only=False)

    if fmt == "auto":
        fmt = "legacy" if "train_graphs" in raw else "spatial"

    if fmt == "legacy":
        print("  Format: legacy (applying lines-as-nodes transform)")
        train_graphs = [lines_as_nodes(g, add_qmax_feature, add_manhole_shortcut, normalize_assets) for g in raw["train_graphs"]]
        val_graphs   = [lines_as_nodes(g, add_qmax_feature, add_manhole_shortcut, normalize_assets) for g in raw["val_graphs"]]
        depth_mean   = float(raw.get("depth_mean", 0.0))
        depth_std    = float(raw.get("depth_std",  1.0))
        for graphs in (train_graphs, val_graphs):
            for g in graphs:
                for nt in g.node_types:
                    x = getattr(g[nt], "x", None)
                    if x is not None and torch.isnan(x).any():
                        torch.nan_to_num_(x, nan=0.0)
        # Feature layout: 0-4 static, 5-10 dynamic depth, 11 sensor_flag
        depth_idxs     = list(range(5, 11))
        sensor_flag_idx = 11
    else:
        print("  Format: spatial (applying lines-as-nodes transform)")
        all_graphs = raw["graphs"]
        depth_mean = float(raw.get("depth_mean", 0.0))
        depth_std  = float(raw.get("depth_std",  1.0))
        if "train_size" in raw:
            n_tr, n_val = int(raw["train_size"]), int(raw["val_size"])
            train_graphs = [lines_as_nodes(g, add_qmax_feature, add_manhole_shortcut, normalize_assets) for g in all_graphs[:n_tr]]
            val_graphs   = [lines_as_nodes(g, add_qmax_feature, add_manhole_shortcut, normalize_assets) for g in all_graphs[n_tr:n_tr + n_val]]
        else:
            n = int(len(all_graphs) * 0.8)
            train_graphs = [lines_as_nodes(g, add_qmax_feature, add_manhole_shortcut, normalize_assets) for g in all_graphs[:n]]
            val_graphs   = [lines_as_nodes(g, add_qmax_feature, add_manhole_shortcut, normalize_assets) for g in all_graphs[n:]]
        for graphs in (train_graphs, val_graphs):
            for g in graphs:
                for nt in g.node_types:
                    x = getattr(g[nt], "x", None)
                    if x is not None and torch.isnan(x).any():
                        torch.nan_to_num_(x, nan=0.0)
                for et in g.edge_types:
                    ea = getattr(g[et], "edge_attr", None)
                    if ea is not None and torch.isnan(ea).any():
                        torch.nan_to_num_(ea, nan=0.0)
        # Feature layout: 0-2 static, 3-N-2 dynamic depth, N-1 sensor_flag
        feat_dim       = train_graphs[0]["manhole"].x.size(1)
        sensor_flag_idx = feat_dim - 1
        depth_idxs     = list(range(3, feat_dim - 1))

    if no_asset_physics:
        for graphs in (train_graphs, val_graphs):
            for g in graphs:
                strip_asset_physics(g)
        kept = {nt: ASSET_GROUP_NUM_SUBTYPES[nt] for nt in ("pipe", "weir", "pump")
                if nt in train_graphs[0].node_types}
        print(f"  ASSET PHYSICS STRIPPED: hydraulic attributes zeroed on "
              f"{sorted(kept)} nodes and on cross-type asset edges; "
              f"only subtype one-hots remain ({kept})")

    if min_mean_depth_m > 0:
        n_before = len(train_graphs)
        train_graphs = filter_dry_snapshots(train_graphs, depth_mean, depth_std, min_mean_depth_m)
        print(f"  Dry-snapshot filter (>{min_mean_depth_m*100:.0f}cm mean): "
              f"{n_before} → {len(train_graphs)} train graphs")

    node_dims, edge_dims, edge_types = schema_from_graph(train_graphs[0])
    return (train_graphs, val_graphs, depth_mean, depth_std,
            node_dims, edge_dims, edge_types, depth_idxs, sensor_flag_idx)


# ── training loop ─────────────────────────────────────────────────────────────

def main(args):
    set_seeds(args.seed)
    # A silent CPU fallback is a 100x slowdown, not a graceful degradation: on a
    # broken node (torch.cuda unavailable through a driver/library mismatch) an
    # epoch went from 502 s to 4259 s, so a 400-epoch run would be killed at the
    # walltime around epoch 60 with an arbitrary checkpoint. --require_gpu makes
    # batch jobs fail immediately instead.
    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable but --require_gpu was set. Refusing to fall back to "
            "CPU: training would be ~8x slower and would not finish. Check the "
            "node's driver/library versions (nvidia-smi).")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    wandb = None
    try:
        import wandb as _wb; _wb.init(project="hydrognn", config=vars(args)); wandb = _wb
    except Exception:
        pass

    print("=" * 70)
    print(f"HydroGNN  |  {Path(args.data_file).stem}  |  device: {device}")
    print("=" * 70)

    # ── load data ──
    print(f"\nLoading {args.data_file} ...")
    (train_graphs, val_graphs, depth_mean, depth_std,
     node_dims, edge_dims, edge_types,
     depth_idxs, sensor_flag_idx) = load_data(Path(args.data_file), args.data_format,
                                               min_mean_depth_m=args.min_mean_depth,
                                               add_qmax_feature=args.add_qmax_feature,
                                               no_asset_physics=args.no_asset_physics,
                                               add_manhole_shortcut=args.manhole_shortcut,
                                               normalize_assets=args.normalize_assets)

    # ── datum guard (hydrognn/datum.py) ──
    # Control-structure thresholds relative to the upstream invert. Must precede
    # patch_invert_units: it relies on crest/switch levels and the invert still
    # sharing one raw unit, and the patch rescales only the invert.
    if args.relative_thresholds:
        cd = None
        if args.threshold_normalise and args.topology_csv:
            cd = chamber_depth_from_topology(REPO / args.topology_csv, num_nodes,
                                             elev_scale=args.threshold_elev_scale)
            print(f"  Chamber depth for threshold normalisation: "
                  f"{'loaded' if cd is not None else 'UNAVAILABLE, using metres'}")
        print("  Relative control thresholds ON (static geometry only, no targets):")
        for gset in (train_graphs, val_graphs):
            relativise_asset_thresholds(gset, elev_scale=args.threshold_elev_scale,
                                        chamber_depth_m=cd,
                                        verbose=(gset is train_graphs))

    if args.datum_patch_units:
        # invert_level_m is exported in feet: dividing by the exact feet-per-metre
        # constant takes negative implied depths from 82.0% to 0.0% with a minimum
        # of +0.01 m. Applied before any mode below so that level - invert is a
        # real depth.
        patch_invert_units([train_graphs, val_graphs])
        print("  Datum: invert_level_m divided by 3.280839895 (feet -> metres)")
    if args.datum_mode != "off":
        enforce_datum(train_graphs, [val_graphs], depth_mean, depth_std,
                      mode=args.datum_mode)

    num_nodes = train_graphs[0]["manhole"].x.size(0)
    print(f"  train={len(train_graphs)}  val={len(val_graphs)}  "
          f"nodes={num_nodes}  depth: μ={depth_mean:.4f}  σ={depth_std:.4f}")
    print(f"  depth feature indices: {depth_idxs}  sensor_flag: {sensor_flag_idx}")

    # Masking protocol: which dynamic columns get zeroed at non-sensor nodes.
    # --mask_current_only hides only the current-depth column (col depth_idxs[0]);
    # depth-history columns remain visible at non-sensor nodes.
    # --mask_depth_cols N hides the first N dynamic columns (N=1 ≡ current-only,
    # N=2 also hides the t-1 depth, N=len(depth_idxs) ≡ full column masking).
    if args.mask_depth_cols is not None and args.mask_depth_cols >= 1:
        masked_depth_idxs = depth_idxs[:args.mask_depth_cols]
    elif args.mask_current_only:
        masked_depth_idxs = depth_idxs[:1]
    else:
        masked_depth_idxs = depth_idxs
    visible = [i for i in depth_idxs if i not in masked_depth_idxs]
    print(f"  MASKING PROTOCOL: hiding cols {masked_depth_idxs} at non-sensor nodes"
          + (f" — history cols {visible} remain visible" if visible else " (all dynamic columns masked)"))

    # ── masking ──
    use_fixed_mask = args.sensor_file is not None
    rng = torch.Generator(); rng.manual_seed(args.seed)
    if use_fixed_mask:
        sensors     = load_sensor_indices(Path(args.sensor_file))
        fixed_mask  = build_fixed_mask(num_nodes, sensors)
        print(f"  Fixed mask: {(~fixed_mask).sum()} sensors / {num_nodes} nodes")
        val_masked  = [apply_mask(g, fixed_mask, masked_depth_idxs, sensor_flag_idx) for g in val_graphs]
        if args.random_train_mask:
            # Training sees a fresh sensor placement for every graph in every
            # epoch, while validation and test keep the real (fixed) placement.
            #
            # With one fixed sensor set the model can learn a positional lookup
            # -- "this manhole usually sits near 1.6 m" -- which is exactly the
            # distance-invariant behaviour measured on all three networks:
            # constant skill at every hop distance, and 20x more sensors buying
            # only 11% accuracy. A lookup does not care where the sensors are.
            # Re-sampling the mask makes every node observed in some epochs and
            # hidden in others, so a per-node table is useless and the model has
            # to infer from whichever neighbours happen to be visible.
            n_s = max(1, int(round(num_nodes * (1.0 - args.mask_rate))))
            print(f"  TRAIN masking: random {args.mask_rate*100:.0f}% "
                  f"({n_s} sensors/graph, re-sampled each epoch); val/test keep the fixed set")
    else:
        n_sensors = max(1, int(round(num_nodes * (1.0 - args.mask_rate))))
        print(f"  Random mask: {args.mask_rate*100:.0f}%  ({n_sensors} sensors/graph, re-sampled each epoch)")
        val_masked = [
            apply_mask(g, build_random_mask(num_nodes, args.mask_rate, rng), masked_depth_idxs, sensor_flag_idx)
            for g in val_graphs
        ]
    # ── sensor context: network-wide summary + per-node climatology ──
    # The climatology is read from the targets, so it must be restricted to the
    # instrumented nodes; see compute_climatology. Without a fixed sensor set there is
    # no stable observed set to restrict it to, so a per-node table is not definable.
    clim_mu = clim_sd = None
    clim_sensor_mask = None
    if args.wetness_features:
        if not args.leaky_climatology:
            if not use_fixed_mask:
                raise ValueError(
                    "--wetness_features needs a fixed sensor set (--sensor_file) so the "
                    "per-node climatology can be restricted to instrumented nodes. With "
                    "random masking the observed set changes every epoch and a per-node "
                    "table is not well defined.")
            clim_sensor_mask = ~fixed_mask
        clim_mu, clim_sd = compute_climatology(train_graphs, sensor_mask=clim_sensor_mask)
        cur_idx = depth_idxs[0]
        val_masked = [add_sensor_context_features(g, clim_mu, clim_sd, cur_idx,
                                                   include_global=args.global_context)
                      for g in val_masked]
        node_dims, edge_dims, edge_types = schema_from_graph(val_masked[0])
        scope = ("ALL nodes (LEAKY: target statistics at unmonitored manholes)"
                 if args.leaky_climatology
                 else f"{int(clim_sensor_mask.sum())} instrumented nodes only, "
                      f"{int((~clim_sensor_mask).sum())} zeroed")
        print(f"  Sensor context features ON: manhole dim → {node_dims['manhole']} "
              f"(+4 network summary, +2 per-node climatology)")
        print(f"    climatology scope: {scope}")

    # ── target reparameterisation ──
    # Applied after clim_mu/clim_sd are computed, so the context channels still
    # carry the true per-node climatology rather than a near-zero anomaly mean.
    target_offset = None
    if args.target_anomaly:
        # The offset is a per-node target mean, so it is restricted to sensors too.
        if not args.leaky_climatology and use_fixed_mask:
            anom_mask = ~fixed_mask
        elif not args.leaky_climatology:
            raise ValueError("--target_anomaly needs a fixed sensor set (--sensor_file) "
                             "unless --leaky_climatology is passed.")
        else:
            anom_mask = None
        target_offset = compute_climatology(train_graphs, sensor_mask=anom_mask)[0].clone()
        for graphs in (train_graphs, val_graphs, val_masked):
            subtract_target_offset(graphs, target_offset)
        print(f"  Target: per-node anomaly about the train-split mean "
              f"(offset |mu| mean {target_offset.abs().mean():.4f} in z-units); "
              f"levels recovered as pred + mu")

    val_loader = DataLoader(val_masked, batch_size=args.batch_size, shuffle=False)

    # ── system-type conditioning ──
    sys_idx_vec: Optional[torch.Tensor] = None
    if args.use_system_type or args.use_type_weighted_loss:
        if not args.topology_csv:
            raise ValueError("--topology_csv is required when using --use_system_type "
                             "or --use_type_weighted_loss")
        sys_idx_vec = load_system_type_idx(Path(args.topology_csv), num_nodes)
        print(f"  System-type conditioning: {sys_idx_vec.unique().tolist()}")

    # per-node L_pred weights by system type (combined/storm/foul/other)
    node_w_vec: Optional[torch.Tensor] = None
    if args.use_type_weighted_loss:
        type_w = torch.tensor([args.w_combined, args.w_storm, args.w_foul, args.w_other])
        node_w_vec = type_w[sys_idx_vec]
        print(f"  Type-weighted L_pred: w={type_w.tolist()}")

    # ── model ──
    model = HydroGNN(
        node_dims=node_dims, edge_dims=edge_dims, edge_types=edge_types,
        hidden_dim=args.hidden_dim, num_blocks=args.num_blocks,
        heads=args.heads, dropout=args.dropout,
        use_softplus_output=args.softplus_output,
        use_system_type_conditioning=args.use_system_type,
        use_input_skip=args.input_skip,
        use_static_highway=args.static_highway,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} params  ({args.num_blocks} blocks, hidden={args.hidden_dim})")

    # Learned task weighting replaces the static lambda_grad / lambda_mass. Its
    # log-variances are registered as a separate param group: they are loss
    # parameters, not network weights, so weight decay must not pull them toward
    # zero variance (that would silently reintroduce a fixed weighting).
    loss_weights = None
    if args.learnable_loss_weights:
        loss_weights = LearnableLossWeights(init_log_var=args.init_log_var).to(device)
        optimizer = optim.AdamW(
            [{"params": model.parameters(), "weight_decay": args.weight_decay},
             {"params": loss_weights.parameters(), "weight_decay": 0.0}],
            lr=args.lr)
        print(f"  Learned loss weighting ON: tasks={loss_weights.tasks} "
              f"init log σ²={args.init_log_var} — lambda_grad/lambda_mass ignored")
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    physics_kw = dict(
        depth_mean=depth_mean, depth_std=depth_std,
        pred_loss=args.pred_loss,
        lambda_pred=args.lambda_pred, lambda_flow=args.lambda_flow,
        lambda_qmax=args.lambda_qmax, lambda_depth=args.lambda_depth,
        lambda_level=args.lambda_level,
        lambda_grad=args.lambda_grad, lambda_mass=args.lambda_mass,
        loss_weights=loss_weights,
        corrected_mass=args.corrected_mass,
    )

    if args.physics_warmup:
        lg1, lm1 = get_physics_weights(
            1, warm_up_epochs=args.physics_warmup_epochs, total_epochs=args.epochs,
            max_lambda_grad=args.lambda_grad or 0.0,
            max_lambda_mass=args.lambda_mass or 0.0, floor=args.physics_warmup_floor)
        print(f"  Physics warmup ON: lambda_grad/lambda_mass cosine-ramped from "
              f"{lg1:.5g}/{lm1:.5g} at epoch 1 to "
              f"{args.lambda_grad or 0.0}/{args.lambda_mass or 0.0} at epoch "
              f"{args.physics_warmup_epochs}, then held")
        # No early stop or checkpoint before the auxiliary terms reach full weight.
        if args.min_epochs < args.physics_warmup_epochs:
            print(f"    min_epochs {args.min_epochs} → {args.physics_warmup_epochs} "
                  f"(no early stop, and no checkpoint written, before physics is at "
                  f"full weight)")
            args.min_epochs = args.physics_warmup_epochs

    best_val_mae = float("inf"); patience = 0
    best_val_mae_post_min = float("inf"); post_min_patience = 0
    print(f"\nTraining for {args.epochs} epochs  (patience={args.patience})\n")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Physics penalties are scheduled per epoch; static weights starve the
        # data term of gradient at high-variance pump/weir nodes early on.
        if args.learnable_loss_weights:
            # Warmup still gates *whether* the physics tasks contribute; their
            # relative weight, once active, is learned rather than scheduled.
            physics_kw["physics_active"] = (
                not args.physics_warmup or epoch >= args.physics_warmup_epochs)
        elif args.physics_warmup:
            lg, lm = get_physics_weights(
                epoch, warm_up_epochs=args.physics_warmup_epochs,
                total_epochs=args.epochs,
                max_lambda_grad=args.lambda_grad or 0.0,
                max_lambda_mass=args.lambda_mass or 0.0,
                floor=args.physics_warmup_floor)
            physics_kw["lambda_grad"] = lg
            physics_kw["lambda_mass"] = lm

        # ── build train epoch ──
        if use_fixed_mask and not args.random_train_mask:
            epoch_graphs = [apply_mask(g, fixed_mask, masked_depth_idxs, sensor_flag_idx)
                            for g in train_graphs]
        else:
            epoch_graphs = [
                apply_mask(g, build_random_mask(num_nodes, args.mask_rate, rng),
                           masked_depth_idxs, sensor_flag_idx)
                for g in train_graphs
            ]
        if args.wetness_features:
            epoch_graphs = [add_sensor_context_features(g, clim_mu, clim_sd, depth_idxs[0],
                                                       include_global=args.global_context)
                            for g in epoch_graphs]
        train_loader = DataLoader(epoch_graphs, batch_size=args.batch_size, shuffle=True)

        # ── train ──
        model.train()
        tr_loss = 0.0; tr_n = 0
        tr_c = {k: 0.0 for k in ("L_pred", "L_flow", "L_qmax", "L_depth", "L_level", "L_grad", "L_mass")}

        for batch in train_loader:
            batch = batch.to(device)
            n_graphs = batch["manhole"].x.size(0) // num_nodes
            if sys_idx_vec is not None:
                batch["manhole"].system_type_idx = sys_idx_vec.to(device).repeat(n_graphs)
            node_w = node_w_vec.to(device).repeat(n_graphs) if node_w_vec is not None else None

            optimizer.zero_grad()
            ea = {et: batch[et].edge_attr if "edge_attr" in batch[et] else None
                  for et in batch.edge_types}
            pred   = model(batch.x_dict, batch.edge_index_dict, ea,
                           system_type_idx=getattr(batch["manhole"], "system_type_idx", None))
            target = batch["manhole"].y.squeeze(-1) if batch["manhole"].y.ndim == 2 \
                     else batch["manhole"].y
            mask   = batch["manhole"].non_sensor_mask
            invert = getattr(batch["manhole"], "invert_level_m", None)

            loss, comps = physics_loss(pred, target, invert,
                                       batch.edge_index_dict, ea,
                                       mask=mask, node_x_dict=batch.x_dict,
                                       node_weights=node_w,
                                       **physics_kw)
            # Opt-in asset-specific hydraulics (separate module, no effect unless
            # --hydraulic_physics is passed). Added on top of the existing terms so
            # this arm can be compared against them directly.
            if args.hydraulic_physics:
                h_loss, h_comps = hydraulic_physics_loss(
                    pred, target, invert, batch.x_dict, batch.edge_index_dict,
                    depth_mean=depth_mean, depth_std=depth_std,
                    target_frame=args.target_frame,
                    elev_scale=(FEET_PER_METRE_TRAIN if args.datum_patch_units else 1.0),
                    lambda_asset=args.lambda_asset,
                    lambda_weir_dry=args.lambda_weir_dry,
                    lambda_bounds=args.lambda_bounds)
                loss = loss + h_loss
                comps.update(h_comps)
            if not torch.isfinite(loss):
                optimizer.zero_grad(); continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item(); tr_n += 1
            for k in tr_c:
                if k in comps:
                    tr_c[k] += comps[k]

        scheduler.step()
        if tr_n:
            tr_loss /= tr_n
            for k in tr_c: tr_c[k] /= tr_n

        # ── val ──
        model.eval()
        v_mae = 0.0; v_n = 0
        v_wet_mae = 0.0; v_wet_n = 0
        WET_THRESH = 0.02  # 2 cm mean physical depth → "wet" snapshot
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                if sys_idx_vec is not None:
                    n_graphs = batch["manhole"].x.size(0) // num_nodes
                    batch["manhole"].system_type_idx = sys_idx_vec.to(device).repeat(n_graphs)
                ea = {et: batch[et].edge_attr if "edge_attr" in batch[et] else None
                      for et in batch.edge_types}
                pred   = model(batch.x_dict, batch.edge_index_dict, ea,
                               system_type_idx=getattr(batch["manhole"], "system_type_idx", None))
                target = batch["manhole"].y.squeeze(-1) if batch["manhole"].y.ndim == 2 \
                         else batch["manhole"].y
                mask   = batch["manhole"].non_sensor_mask
                v_mae += float((pred - target).abs()[mask].mean()) * depth_std * 100.0
                v_n   += 1

                # Per-graph wet-only MAE using batch index
                b_idx = batch["manhole"].batch if hasattr(batch["manhole"], "batch") \
                        and batch["manhole"].batch is not None else torch.zeros(
                            target.size(0), dtype=torch.long, device=target.device)
                n_graphs_batch = int(b_idx.max().item()) + 1
                for gi in range(n_graphs_batch):
                    gm = b_idx == gi
                    tg = target[gm]
                    if target_offset is not None:
                        tg = tg + target_offset.to(tg.device)
                    mean_phys = float(tg.mean()) * depth_std + depth_mean
                    if mean_phys >= WET_THRESH:
                        gm_ns = gm & mask
                        if gm_ns.any():
                            v_wet_mae += float((pred - target).abs()[gm_ns].mean()) * depth_std * 100.0
                            v_wet_n += 1
        val_mae_cm     = v_mae / max(v_n, 1)
        val_wet_mae_cm = v_wet_mae / max(v_wet_n, 1) if v_wet_n > 0 else float("nan")

        # ── checkpoint ──
        is_best = val_mae_cm < best_val_mae
        if is_best:
            best_val_mae = val_mae_cm; patience = 0
        else:
            patience += 1

        # When --min_epochs is set, best_model.pth is only written from min_epochs
        # onward, using a separate patience counter that starts fresh at min_epochs.
        # Before min_epochs the global patience counter still prevents infinite stalls.
        past_min = (epoch >= args.min_epochs)
        if past_min:
            if val_mae_cm < best_val_mae_post_min:
                best_val_mae_post_min = val_mae_cm; post_min_patience = 0
                torch.save({
                    "epoch": epoch, "model_state_dict": model.state_dict(),
                    "node_dims": node_dims, "edge_dims": edge_dims, "edge_types": edge_types,
                    "depth_mean": depth_mean, "depth_std": depth_std,
                    "target_offset": target_offset,
                    "depth_idxs": depth_idxs, "sensor_flag_idx": sensor_flag_idx,
                    "val_mae_cm": val_mae_cm, "args": vars(args),
                }, out_dir / "best_model.pth")
            else:
                post_min_patience += 1
        elif is_best:
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "node_dims": node_dims, "edge_dims": edge_dims, "edge_types": edge_types,
                "depth_mean": depth_mean, "depth_std": depth_std,
                "depth_idxs": depth_idxs, "sensor_flag_idx": sensor_flag_idx,
                "val_mae_cm": val_mae_cm, "args": vars(args),
            }, out_dir / "best_model_pre_min.pth")

        wet_str = f"{val_wet_mae_cm:.2f}" if v_wet_n > 0 else "  -- "
        best_marker = "*" if (past_min and val_mae_cm == best_val_mae_post_min) or \
                             (not past_min and is_best) else " "
        print(
            f"[{epoch:3d}/{args.epochs}] "
            f"loss={tr_loss:.4f}  val_mae={val_mae_cm:.2f}cm{best_marker}  "
            f"wet={wet_str}cm({v_wet_n})  "
            f"Lp={tr_c['L_pred']:.4f} Lf={tr_c['L_flow']:.4f} "
            f"Lq={tr_c['L_qmax']:.4f} Ld={tr_c['L_depth']:.4f} Ll={tr_c['L_level']:.4f}"
            + (f" Lg={tr_c['L_grad']:.4f}" if tr_c['L_grad'] > 0 else "")
            + (f" Lm={tr_c['L_mass']:.4f}" if tr_c['L_mass'] > 0 else "")
            + f"  {time.time()-t0:.1f}s", flush=True,
        )
        if wandb:
            wandb.log({"epoch": epoch, "train/loss": tr_loss, "val/mae_cm": val_mae_cm,
                       "val/wet_mae_cm": val_wet_mae_cm,
                       **{f"train/{k}": v for k, v in tr_c.items()},
                       "lr": optimizer.param_groups[0]["lr"]})

        # Early stopping only counts epochs after min_epochs.
        if past_min and post_min_patience >= args.patience:
            print(f"Early stopping at epoch {epoch} "
                  f"({args.patience} epochs without improvement since "
                  f"min_epochs={args.min_epochs})"); break

    print(f"\nDone. Best val MAE = {best_val_mae:.2f} cm  →  {out_dir / 'best_model.pth'}")

    score_test_split(args, model, out_dir, device, num_nodes, depth_std,
                     depth_idxs, sensor_flag_idx, fixed_mask if use_fixed_mask else None,
                     clim_mu, clim_sd, sys_idx_vec,
                     target_offset=target_offset,
                     depth_mean_for_test=depth_mean)


def score_test_split(args, model, out_dir, device, num_nodes, depth_std,
                     depth_idxs, sensor_flag_idx, fixed_mask,
                     clim_mu, clim_sd, sys_idx_vec,
                     target_offset=None, depth_mean_for_test=0.0) -> None:
    """Score the held-out test split once, from the best-validation checkpoint.

    Selection has already finished at this point: the test graphs took no part in
    early stopping or in choosing the checkpoint, so this is a clean
    generalisation estimate. If the dataset carries no test split we write
    nothing — falling back to the validation graphs would silently relabel a
    selection metric as a test metric.
    """
    raw = torch.load(args.data_file, map_location="cpu", weights_only=False)
    if "graphs" not in raw or "train_size" not in raw:
        print("Test scoring skipped: dataset has no spatial train/val/test layout.")
        return
    n_tr, n_val = int(raw["train_size"]), int(raw["val_size"])
    n_test = int(raw.get("test_size", len(raw["graphs"]) - n_tr - n_val))
    if n_test <= 0:
        print("Test scoring skipped: dataset contains no held-out test split. "
              "Refusing to fall back to the validation split.")
        return

    ckpt_path = out_dir / "best_model.pth"
    if not ckpt_path.exists():
        print(f"Test scoring skipped: {ckpt_path} not found.")
        return
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    print(f"\nScoring held-out TEST split ({n_test} snapshots) from best-val "
          f"checkpoint (epoch {ck['epoch']}, val MAE {ck['val_mae_cm']:.2f} cm)...")

    test_graphs = [lines_as_nodes(g, args.add_qmax_feature, args.manhole_shortcut,
                                 args.normalize_assets)
                   for g in raw["graphs"][n_tr + n_val: n_tr + n_val + n_test]]
    for g in test_graphs:
        for nt in g.node_types:
            x = getattr(g[nt], "x", None)
            if x is not None and torch.isnan(x).any():
                torch.nan_to_num_(x, nan=0.0)
        for et in g.edge_types:
            ea = getattr(g[et], "edge_attr", None)
            if ea is not None and torch.isnan(ea).any():
                torch.nan_to_num_(ea, nan=0.0)
    if args.no_asset_physics:
        for g in test_graphs:
            strip_asset_physics(g)
    if target_offset is not None:
        # same shift the model was trained under; MAE/RMSE are unchanged by it
        subtract_target_offset(test_graphs, target_offset)
    # The datum transform is applied in main() to the train/val graphs only, but
    # these test graphs are rebuilt from the raw file, so it must be repeated
    # here. Without it a model trained to predict depth is scored against level
    # targets, which on Heusden is a ~149 cm constant offset.
    if getattr(args, "datum_patch_units", False):
        patch_invert_units([test_graphs])
    if getattr(args, "datum_mode", "off") not in ("off", "check"):
        enforce_datum(test_graphs, [], depth_mean_for_test, depth_std,
                      mode=args.datum_mode, verbose=False)

    mask = fixed_mask if fixed_mask is not None else \
        build_random_mask(num_nodes, args.mask_rate, torch.Generator().manual_seed(args.seed))
    masked_depth_idxs = depth_idxs if not args.mask_current_only else depth_idxs[:1]
    if args.mask_depth_cols is not None and args.mask_depth_cols >= 1:
        masked_depth_idxs = depth_idxs[:args.mask_depth_cols]
    test_masked = [apply_mask(g, mask, masked_depth_idxs, sensor_flag_idx) for g in test_graphs]
    if args.wetness_features:
        test_masked = [add_sensor_context_features(g, clim_mu, clim_sd, depth_idxs[0],
                                                  include_global=args.global_context)
                       for g in test_masked]

    # Control-adjacent manholes: endpoints of a pump or weir asset. These are the
    # sharp spatial discontinuities the heterogeneous relations exist to model.
    ep = asset_endpoint_indices(test_masked[0])
    ctrl = torch.zeros(num_nodes, dtype=torch.bool)
    for grp in ("weir", "pump"):
        if grp in ep:
            us, ds = ep[grp]
            ctrl[us.long()] = True
            ctrl[ds.long()] = True

    err_sum = torch.zeros(num_nodes)
    sq_sum = torch.zeros(num_nodes)
    n_obs = 0
    with torch.no_grad():
        for batch in DataLoader(test_masked, batch_size=args.batch_size, shuffle=False):
            batch = batch.to(device)
            ng = batch["manhole"].x.size(0) // num_nodes
            sti = sys_idx_vec.to(device).repeat(ng) if sys_idx_vec is not None else None
            ea = {et: batch[et].edge_attr if "edge_attr" in batch[et] else None
                  for et in batch.edge_types}
            pred = model(batch.x_dict, batch.edge_index_dict, ea, system_type_idx=sti)
            y = batch["manhole"].y
            y = y.squeeze(-1) if y.ndim == 2 else y
            d = ((pred - y) * depth_std * 100.0).view(ng, num_nodes).cpu()
            err_sum += d.abs().sum(0)
            sq_sum += (d ** 2).sum(0)
            n_obs += ng

    nsm = mask.cpu()                              # True = non-sensor (scored)
    def grp_metrics(sel):
        sel = sel & nsm
        n = int(sel.sum())
        if n == 0:
            return None
        return {"n_nodes": n,
                "mae_cm": float(err_sum[sel].sum() / (n * n_obs)),
                "rmse_cm": float((sq_sum[sel].sum() / (n * n_obs)).sqrt())}

    metrics = {
        "n_test_snapshots": n_obs,
        "best_val_mae_cm": float(ck["val_mae_cm"]),
        "best_epoch": int(ck["epoch"]),
        "overall": grp_metrics(torch.ones(num_nodes, dtype=torch.bool)),
        "control_adjacent": grp_metrics(ctrl),
        "pipe_only": grp_metrics(~ctrl),
    }
    if "fold" in raw:
        metrics["fold"] = int(raw["fold"])
        metrics["fold_assignment"] = raw.get("fold_assignment")

    import json
    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
    o, a, b = metrics["overall"], metrics["control_adjacent"], metrics["pipe_only"]
    print(f"  TEST overall          MAE={o['mae_cm']:7.3f} cm  RMSE={o['rmse_cm']:7.3f} cm  (n={o['n_nodes']})")
    if a:
        print(f"  TEST control-adjacent MAE={a['mae_cm']:7.3f} cm  RMSE={a['rmse_cm']:7.3f} cm  (n={a['n_nodes']})")
    if b:
        print(f"  TEST pipe-only        MAE={b['mae_cm']:7.3f} cm  RMSE={b['rmse_cm']:7.3f} cm  (n={b['n_nodes']})")
    print(f"  → {out_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HydroGNN unified training")
    # data
    ap.add_argument("--data_file",    required=True)
    ap.add_argument("--data_format",  default="auto", choices=["auto", "spatial", "legacy"])
    ap.add_argument("--sensor_file",  default=None, help="Fixed sensor file (one integer index per line)")
    ap.add_argument("--topology_csv", default=None, help="Node CSV for system_type (needed with --use_system_type)")
    ap.add_argument("--output_dir",   required=True)
    # model
    ap.add_argument("--hidden_dim",      type=int,   default=128)
    ap.add_argument("--num_blocks",      type=int,   default=4)
    ap.add_argument("--heads",           type=int,   default=4)
    ap.add_argument("--dropout",         type=float, default=0.1)
    ap.add_argument("--use_system_type", action="store_true",
                    help="Enable per-node sewer system type FiLM conditioning")
    ap.add_argument("--softplus_output", action="store_true",
                    help="Apply Softplus activation to depth head output (non-negative predictions). "
                         "Disabled by default.")
    ap.add_argument("--min_epochs", type=int, default=0,
                    help="Minimum epochs before best_model.pth is written and patience is "
                         "counted. Pre-min checkpoints go to best_model_pre_min.pth. "
                         "Default 0 restores original behaviour.")
    ap.add_argument("--add_qmax_feature", action="store_true",
                    help="Append log1p(Manning Q_max) to pipe node features (dim 6 → 7)")
    ap.add_argument("--no_asset_physics", action="store_true",
                    help="Ablation: zero all hydraulic attributes on pipe/weir/pump asset "
                         "nodes and on cross-type asset edges, keeping only subtype "
                         "one-hots. Feature dims and parameter count are unchanged.")
    ap.add_argument("--input_skip", action="store_true",
                    help="Concatenate raw manhole input features to the final embedding "
                         "before the depth head, so input signals reach the head without "
                         "passing through all message-passing blocks.")
    ap.add_argument("--wetness_features", action="store_true",
                    help="Append sensor context features to each manhole node: 4 network-wide "
                         "summary statistics (mean/max/p90/std of sensor depths per snapshot) "
                         "+ 2 per-node climatology statistics (training-period mean/std). "
                         "The climatology is restricted to instrumented nodes and zeroed "
                         "elsewhere unless --leaky_climatology is passed.")
    ap.add_argument("--leaky_climatology", action="store_true",
                    help="Compute the per-node climatology channels at every manhole instead "
                         "of only at sensors. Uses target statistics at unmonitored nodes; "
                         "not used in the paper.")
    # training
    ap.add_argument("--epochs",       type=int,   default=400)
    ap.add_argument("--batch_size",   type=int,   default=16)
    ap.add_argument("--lr",           type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-5)
    ap.add_argument("--mask_rate",       type=float, default=0.95)
    ap.add_argument("--require_gpu", action="store_true",
                    help="Abort if CUDA is unavailable rather than silently training "
                         "on CPU. Use for every batch job.")
    ap.add_argument("--datum_patch_units", action="store_true",
                    help="Divide invert_level_m by 3.280839895 (feet to metres) "
                         "before the datum mode runs. Note this cannot change "
                         "what the network learns: the invert feature column is "
                         "z-scored and the raw attribute reaches only the physics "
                         "loss, which is zero-weighted during warmup. It matters "
                         "for anything computing level - invert.")
    ap.add_argument("--datum_mode", default="check",
                    choices=["check", "align", "depth", "patch_units", "off"],
                    help="Reconcile the target datum with invert_level_m. "
                         "'check' fails when level-invert is negative for more "
                         "than 2 percent of node-snapshots (Heusden: 83.9); "
                         "'align' rewrites invert onto the target datum from a "
                         "train-split fit; 'depth' converts the target to depth "
                         "above invert clamped at 0; 'off' skips the guard.")
    ap.add_argument("--global_context", action="store_true",
                    help="Append the 4 network-wide sensor order statistics to "
                         "every manhole.")
    ap.add_argument("--static_highway", action="store_true",
                    help="Concatenate the raw [invert, x, y] vector straight to "
                         "the readout, bypassing all message-passing blocks.")
    ap.add_argument("--target_anomaly", action="store_true",
                    help="Predict the per-node anomaly about the train-split mean "
                         "level instead of the absolute level. Removes the static "
                         "elevation component from the learning problem; MAE and "
                         "RMSE are unchanged in definition.")
    ap.add_argument("--normalize_assets", action="store_true",
                    help="Z-normalise each asset group's hydraulic columns. Without it the "
                         "pipe block reaches absmax 454 against 5.2 for the z-normalised "
                         "manhole block, swamping the hydraulic signal.")
    ap.add_argument("--manhole_shortcut", action="store_true",
                    help="Add (manhole, adjacent, manhole) edges alongside the asset "
                         "nodes so an adjacent reading is copyable in one hop. Keeps the "
                         "lines-as-nodes graph fully intact; also doubles manhole reach.")
    ap.add_argument("--random_train_mask", action="store_true",
                    help="Re-sample the sensor placement every graph every epoch during "
                         "TRAINING while val/test keep --sensor_file. Stops the model "
                         "learning a per-node lookup tied to one fixed sensor set.")
    ap.add_argument("--mask_current_only", action="store_true",
                    help="Hide only the current-depth column at non-sensor nodes; "
                         "depth-history columns remain visible. Default masks all "
                         "dynamic depth columns.")
    ap.add_argument("--mask_depth_cols", type=int, default=None,
                    help="Hide the first N dynamic depth columns at non-sensor nodes "
                         "(1=current only, 2=current+t-1, omit=all). Overrides "
                         "--mask_current_only when set.")
    ap.add_argument("--patience",        type=int,   default=15)
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--min_mean_depth",  type=float, default=0.0,
                    help="Skip training snapshots where mean manhole depth < this value (metres). "
                         "Filters near-dry snapshots with no learning signal.")
    # loss configuration
    ap.add_argument("--pred_loss", default="mse", choices=["mse", "mae"],
                    help="Error metric for the prediction loss term L_pred.")
    ap.add_argument("--use_type_weighted_loss", action="store_true",
                    help="Weight L_pred per node by sewer system type (needs --topology_csv)")
    ap.add_argument("--w_combined", type=float, default=2.0)
    ap.add_argument("--w_storm",    type=float, default=1.2)
    ap.add_argument("--w_foul",     type=float, default=1.0)
    ap.add_argument("--w_other",    type=float, default=1.5)
    # physics loss weights
    ap.add_argument("--lambda_pred",  type=float, default=1.0)
    ap.add_argument("--lambda_flow",  type=float, default=0.1)
    ap.add_argument("--lambda_qmax",  type=float, default=0.01)
    ap.add_argument("--lambda_depth", type=float, default=0.1)
    ap.add_argument("--lambda_level", type=float, default=0.1)
    ap.add_argument("--lambda_grad",  type=float, default=None,
                    help="Weight for spatial depth-gradient matching loss L_grad.")
    ap.add_argument("--lambda_mass",  type=float, default=None,
                    help="Weight for z-normalised mass conservation loss L_mass.")
    ap.add_argument("--corrected_mass", action="store_true",
                    help="Use the residual-matching mass balance (water levels, non-zero "
                         "target residual) instead of the degenerate zero-residual form.")
    ap.add_argument("--learnable_loss_weights", action="store_true",
                    help="Learn task variances for the data/mass/grad terms instead of "
                         "using fixed lambda_grad/lambda_mass (Kendall-style weighting).")
    ap.add_argument("--init_log_var", type=float, default=0.0,
                    help="Initial log sigma^2 for every learned task weight.")
    ap.add_argument("--physics_warmup", action="store_true",
                    help="Cosine-ramp lambda_grad/lambda_mass from the first epoch up to their "
                         "configured values, reaching full weight at --physics_warmup_epochs.")
    ap.add_argument("--physics_warmup_epochs", type=int, default=30,
                    help="Epochs taken to REACH full physics weight (with --physics_warmup). "
                         "Also raises --min_epochs to this value.")
    ap.add_argument("--relative_thresholds", action="store_true",
                    help="Rewrite weir crest and pump switch levels as a depth above "
                         "the UPSTREAM manhole's own invert, in metres, instead of an "
                         "absolute datum elevation in the export's raw unit. Static "
                         "geometry only -- no targets, no snapshots.")
    ap.add_argument("--threshold_elev_scale", type=float, default=3.280839895013123,
                    help="Raw elevation units per metre for --relative_thresholds "
                         "(3.2808 for the Heusden feet export, 1.0 for metric SWMM).")
    ap.add_argument("--threshold_normalise", action="store_true",
                    help="Also divide the relative threshold by the upstream manhole's "
                         "chamber depth, giving a dimensionless fraction-of-chamber.")
    ap.add_argument("--virtual_node", action="store_true",
                    help="Add a master node wired to every manhole, updated and "
                         "broadcast after each block, so any node is 2 hops from any "
                         "sensor. No per-node parameters, so unlike --global_lowrank it "
                         "stays transferable to an unseen network.")
    ap.add_argument("--global_lowrank", action="store_true",
                    help="Add the learned low-rank global pathway: sensor readings are "
                         "pooled to k coefficients that reach every node in one step, "
                         "instead of propagating hop by hop. At 1%% coverage the median "
                         "manhole is 9 hops from a sensor and 4 blocks reach 2.")
    ap.add_argument("--global_lowrank_k", type=int, default=16,
                    help="Rank of the global pathway (number of learned modes).")
    ap.add_argument("--hydraulic_physics", action="store_true",
                    help="Enable the asset-specific hydraulic terms in "
                         "hydrognn/physics_hydraulic.py (weir 3/2 crest law, pump "
                         "switching threshold, physical bounds). Off by default; the "
                         "existing lambda_grad/lambda_mass terms are unaffected.")
    ap.add_argument("--target_frame", default="level", choices=["level", "depth"],
                    help="'level' = target is water level above datum (InfoWorks "
                         "Heusden); 'depth' = target is depth above invert (SWMM "
                         "Bellinge/Tuindorp). Sets whether the invert is added when "
                         "forming physical levels for the hydraulic laws.")
    ap.add_argument("--lambda_asset", type=float, default=0.0,
                    help="Weight on the per-asset-class continuity residual "
                         "(weir and pump laws, normalised within class).")
    ap.add_argument("--lambda_weir_dry", type=float, default=0.0,
                    help="Weight on the unsupervised dry-weir consistency term: a "
                         "weir below its crest must not drive a downstream rise.")
    ap.add_argument("--lambda_bounds", type=float, default=0.0,
                    help="Weight on the unsupervised bound constraint "
                         "(invert <= level, and level <= ground where available).")
    ap.add_argument("--physics_warmup_floor", type=float, default=0.0,
                    help="Fraction of the full physics weight applied at epoch 1 (e.g. 1e-3 for "
                         "a small non-zero nudge from the first batch). Default 0.0.")
    ap.add_argument("--lambda_bound",         type=float, default=None)
    ap.add_argument("--headroom_znorm",       type=float, default=None)
    ap.add_argument("--grad_tolerance_znorm", type=float, default=None)
    main(ap.parse_args())
