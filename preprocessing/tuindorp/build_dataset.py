#!/usr/bin/env python3
"""Build the Tuindorp dataset from the 4TU simulations.

Output: data/processed/tuindorp_v2.pt (then run attach_inflow.py)

Node features (10):
  [0] ground_level  invert + max depth [m]
  [1] x_coord
  [2] y_coord
  [3] depth_current depth above invert at t
  [4] temp_recent   depth at t-1
  [5] temp_mean     mean depth over the last W steps
  [6] temp_max      max depth over the last W steps
  [7] temp_velocity depth[t] - depth[t-1]
  [8] temp_trend    (depth[t] - depth[t-W]) / W
  [9] sensor_flag   set during training
All features are z-normalised.

Edge attributes (4): diameter [m], length / 100, manning_n, slope (0).
"""
from __future__ import annotations
import argparse
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data/raw/tuindorp/Tuindorp development - 1 min resolution"
NET_DIR   = DATA_ROOT / "networks"
SIM_ROOT  = DATA_ROOT / "simulations"

NODE_CSV    = NET_DIR / "Tuindorp development - 1 min resolution_node.csv"
CONDUIT_CSV = NET_DIR / "Tuindorp development - 1 min resolution_conduit.csv"

TEMPORAL_WINDOW = 12   # look-back steps for temporal features
SAMPLES_PER_SIM = 15   # random timesteps drawn from each simulation
WARMUP          = TEMPORAL_WINDOW  # skip first W steps (no history)
SEED            = 42

# ── topology ──────────────────────────────────────────────────────────────────

def load_topology():
    nodes_df    = pd.read_csv(NODE_CSV)
    conduits_df = pd.read_csv(CONDUIT_CSV)
    return nodes_df, conduits_df


def build_index_maps(nodes_df):
    """Separate manholes (Junction) from outfalls, return ordered id lists and maps."""
    manholes = nodes_df[nodes_df["node_type"] == "Junction"].reset_index(drop=True)
    outfalls  = nodes_df[nodes_df["node_type"] == "Outfall"].reset_index(drop=True)
    mh_id_to_idx = {nid: i for i, nid in enumerate(manholes["node_id"])}
    of_id_to_idx = {nid: i for i, nid in enumerate(outfalls["node_id"])}
    return manholes, outfalls, mh_id_to_idx, of_id_to_idx


def build_static_features(manholes, outfalls):
    """Compute static (non-temporal) features: ground_level, x, y."""
    mh_static = np.stack([
        manholes["elevation"].values + manholes["max_depth"].values,  # ground_level
        manholes["x"].values,
        manholes["y"].values,
    ], axis=1).astype(np.float32)  # [N_mh, 3]

    of_static = np.stack([
        outfalls["elevation"].values + outfalls["max_depth"].values,
        outfalls["x"].values,
        outfalls["y"].values,
    ], axis=1).astype(np.float32)  # [N_of, 3]

    return mh_static, of_static


def build_edge_index_and_attr(conduits_df, mh_id_to_idx, of_id_to_idx):
    """Build edge_index and edge_attr tensors for (manhole,conduit,manhole) and
    (manhole,conduit,outfall) edge types."""
    mh_mh_src, mh_mh_dst, mh_mh_attr = [], [], []
    mh_of_src, mh_of_dst, mh_of_attr = [], [], []

    for _, row in conduits_df.iterrows():
        us = str(row["upstream_node"])
        ds = str(row["downstream_node"])
        # edge attributes: [diameter_m, length/100, manning_n, slope=0]
        dia = float(row["diameter"]) if not pd.isna(row["diameter"]) else 0.3
        lgt = float(row["length"])
        mng = float(row["manning"]) if not pd.isna(row["manning"]) else 0.013
        attr = [dia, lgt / 100.0, mng, 0.0]

        us_mh = mh_id_to_idx.get(us)
        ds_mh = mh_id_to_idx.get(ds)
        ds_of = of_id_to_idx.get(ds)

        if us_mh is not None and ds_mh is not None:
            mh_mh_src.append(us_mh); mh_mh_dst.append(ds_mh); mh_mh_attr.append(attr)
        elif us_mh is not None and ds_of is not None:
            mh_of_src.append(us_mh); mh_of_dst.append(ds_of); mh_of_attr.append(attr)

    def to_ei(src, dst): return torch.tensor([src, dst], dtype=torch.long)
    def to_ea(attr):     return torch.tensor(attr, dtype=torch.float32)

    mh_mh_ei = to_ei(mh_mh_src, mh_mh_dst) if mh_mh_src else torch.zeros(2, 0, dtype=torch.long)
    mh_mh_ea = to_ea(mh_mh_attr)            if mh_mh_attr else torch.zeros(0, 4)
    mh_of_ei = to_ei(mh_of_src, mh_of_dst) if mh_of_src else torch.zeros(2, 0, dtype=torch.long)
    mh_of_ea = to_ea(mh_of_attr)            if mh_of_attr else torch.zeros(0, 4)

    return mh_mh_ei, mh_mh_ea, mh_of_ei, mh_of_ea


# ── simulation loading ─────────────────────────────────────────────────────────

def load_sim_depths(sim_dir: Path, manholes, mh_id_to_idx) -> Optional[np.ndarray]:
    """Load hydraulic_head.csv and convert to depth above invert [m].
    Returns array [T, N_mh] or None if loading fails.
    """
    hh_path = sim_dir / "hydraulic_head.csv"
    if not hh_path.exists():
        return None
    try:
        df = pd.read_csv(hh_path, index_col=0)
    except Exception:
        return None

    N_mh = len(manholes)
    invert = manholes["elevation"].values  # [N_mh]

    # Map column names → manhole index
    depth_arr = np.full((len(df), N_mh), np.nan, dtype=np.float32)
    for col in df.columns:
        # Column format: "node_j_9006F_Hydraulic_head"
        node_id = col.replace("node_", "").replace("_Hydraulic_head", "")
        idx = mh_id_to_idx.get(node_id)
        if idx is not None:
            head = df[col].values.astype(np.float32)
            depth_arr[:, idx] = np.maximum(head - invert[idx], 0.0)

    # Drop rows with any NaN
    valid = ~np.isnan(depth_arr).any(axis=1)
    depth_arr = depth_arr[valid]
    return depth_arr if len(depth_arr) > WARMUP + 1 else None


def temporal_features(depth: np.ndarray, t: int, W: int = TEMPORAL_WINDOW) -> np.ndarray:
    """Compute temporal features at timestep t.
    Returns [N_mh, 5]: [recent, mean, max, velocity, trend]
    """
    t0 = max(0, t - W)
    window = depth[t0:t]          # [min(W,t), N_mh]

    recent   = depth[t - 1] if t > 0 else depth[t]
    mean_d   = window.mean(axis=0) if len(window) > 0 else depth[t]
    max_d    = window.max(axis=0)  if len(window) > 0 else depth[t]
    vel      = depth[t] - recent
    trend    = (depth[t] - depth[t0]) / max(t - t0, 1)
    return np.stack([recent, mean_d, max_d, vel, trend], axis=1)  # [N_mh, 5]


# ── graph construction ─────────────────────────────────────────────────────────

def build_graph(
    t: int,
    depth_arr: np.ndarray,
    mh_static: np.ndarray,
    of_static: np.ndarray,
    mh_mh_ei, mh_mh_ea,
    mh_of_ei, mh_of_ea,
    manholes,
) -> HeteroData:
    """Build one HeteroData graph for timestep t."""
    N_mh = mh_static.shape[0]
    d_t = depth_arr[t]           # [N_mh] raw depth [m]
    tf  = temporal_features(depth_arr, t)  # [N_mh, 5]

    # Stack 10 features (un-normalised — normalisation happens after global stats)
    x_mh = np.concatenate([
        mh_static,                   # [N_mh, 3]: ground_level, x, y
        d_t[:, None],                # [N_mh, 1]: depth_current
        tf,                          # [N_mh, 5]: temporal
        np.ones((N_mh, 1)),          # [N_mh, 1]: sensor_flag = 1.0
    ], axis=1).astype(np.float32)    # [N_mh, 10]

    g = HeteroData()

    g["manhole"].x = torch.from_numpy(x_mh)
    g["manhole"].y = torch.from_numpy(d_t).unsqueeze(1)  # target [N_mh, 1]
    g["manhole"].invert_level_m = torch.from_numpy(
        manholes["elevation"].values.astype(np.float32))

    g["outfall"].x = torch.from_numpy(of_static)

    if mh_mh_ei.size(1) > 0:
        g["manhole", "conduit", "manhole"].edge_index = mh_mh_ei
        g["manhole", "conduit", "manhole"].edge_attr  = mh_mh_ea
    if mh_of_ei.size(1) > 0:
        g["manhole", "conduit", "outfall"].edge_index = mh_of_ei
        g["manhole", "conduit", "outfall"].edge_attr  = mh_of_ea

    return g


# ── normalisation ─────────────────────────────────────────────────────────────

def compute_z_stats(graphs: List[HeteroData]) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Compute dataset-level z-norm stats from all graphs in the training split."""
    all_depths = []
    all_gl, all_x, all_y = [], [], []
    for g in graphs:
        x = g["manhole"].x.numpy()
        all_depths.append(x[:, 3])
        all_gl.append(x[:, 0])
        all_x.append(x[:, 1])
        all_y.append(x[:, 2])
    depths = np.concatenate(all_depths)
    depth_mean = float(depths.mean())
    depth_std  = float(depths.std()) or 1.0
    # Static feature stats for normalising columns 0-2
    feat_means = np.array([
        np.concatenate(all_gl).mean(),
        np.concatenate(all_x).mean(),
        np.concatenate(all_y).mean(),
    ])
    feat_stds = np.array([
        np.concatenate(all_gl).std() or 1.0,
        np.concatenate(all_x).std() or 1.0,
        np.concatenate(all_y).std() or 1.0,
    ])
    return depth_mean, depth_std, feat_means, feat_stds


def normalise_graphs(graphs: List[HeteroData], depth_mean: float, depth_std: float,
                     feat_means: np.ndarray, feat_stds: np.ndarray):
    """Z-normalise features in-place (matches Bellinge format)."""
    for g in graphs:
        x = g["manhole"].x
        # Cols 0-2: static spatial (normalise separately)
        x[:, 0] = (x[:, 0] - feat_means[0]) / feat_stds[0]
        x[:, 1] = (x[:, 1] - feat_means[1]) / feat_stds[1]
        x[:, 2] = (x[:, 2] - feat_means[2]) / feat_stds[2]
        # Cols 3-8: depth and temporal (all normalised with depth stats)
        x[:, 3:9] = (x[:, 3:9] - depth_mean) / depth_std
        # Col 9: sensor_flag stays 1.0
        g["manhole"].x = x
        # Target: z-normalise depth
        g["manhole"].y = (g["manhole"].y - depth_mean) / depth_std


# ── split loading ─────────────────────────────────────────────────────────────

def load_split(split: str, mh_static, of_static, mh_mh_ei, mh_mh_ea, mh_of_ei,
               mh_of_ea, manholes, mh_id_to_idx, rng, n_samples: int = SAMPLES_PER_SIM
               ) -> List[HeteroData]:
    split_dir = SIM_ROOT / split
    if not split_dir.exists():
        print(f"  [warn] split dir not found: {split_dir}")
        return []

    graphs = []
    sims = sorted(split_dir.iterdir())
    for sim_dir in sims:
        if not sim_dir.is_dir():
            continue
        depth_arr = load_sim_depths(sim_dir, manholes, mh_id_to_idx)
        if depth_arr is None:
            continue
        T = len(depth_arr)
        eligible = list(range(WARMUP, T))
        chosen = rng.sample(eligible, min(n_samples, len(eligible)))
        for t in chosen:
            g = build_graph(t, depth_arr, mh_static, of_static,
                            mh_mh_ei, mh_mh_ea, mh_of_ei, mh_of_ea, manholes)
            graphs.append(g)

    print(f"  {split}: {len(sims)} sims → {len(graphs)} graphs")
    return graphs


# ── main ──────────────────────────────────────────────────────────────────────

def main(args):
    random.seed(SEED)
    rng = random.Random(SEED)

    out_dir = REPO / "data/processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tuindorp_v2.pt"

    print("Loading topology…")
    nodes_df, conduits_df = load_topology()
    manholes, outfalls, mh_id_to_idx, of_id_to_idx = build_index_maps(nodes_df)
    mh_static, of_static = build_static_features(manholes, outfalls)
    mh_mh_ei, mh_mh_ea, mh_of_ei, mh_of_ea = build_edge_index_and_attr(
        conduits_df, mh_id_to_idx, of_id_to_idx)

    print(f"  Manholes: {len(manholes)}, Outfalls: {len(outfalls)}")
    print(f"  (mh,conduit,mh) edges: {mh_mh_ei.size(1)}")
    print(f"  (mh,conduit,outfall) edges: {mh_of_ei.size(1)}")

    print("\nLoading simulations…")
    train_graphs = load_split("training",   mh_static, of_static, mh_mh_ei, mh_mh_ea,
                              mh_of_ei, mh_of_ea, manholes, mh_id_to_idx, rng)
    val_graphs   = load_split("validation", mh_static, of_static, mh_mh_ei, mh_mh_ea,
                              mh_of_ei, mh_of_ea, manholes, mh_id_to_idx, rng)
    test_graphs  = load_split("testing",    mh_static, of_static, mh_mh_ei, mh_mh_ea,
                              mh_of_ei, mh_of_ea, manholes, mh_id_to_idx, rng)

    all_graphs = train_graphs + val_graphs + test_graphs
    if not all_graphs:
        raise RuntimeError("No graphs loaded — check data paths")

    print("\nComputing normalisation stats from training split…")
    depth_mean, depth_std, feat_means, feat_stds = compute_z_stats(train_graphs)
    print(f"  depth_mean={depth_mean:.4f}  depth_std={depth_std:.4f}")

    print("Normalising all graphs…")
    normalise_graphs(all_graphs, depth_mean, depth_std, feat_means, feat_stds)

    # Derive schema from first graph
    g0 = all_graphs[0]
    node_dims = {nt: g0[nt].x.size(-1) for nt in g0.node_types if hasattr(g0[nt], 'x')}
    edge_dims  = {}
    edge_types = []
    for et in g0.edge_types:
        edge_types.append(et)
        ea = g0[et].edge_attr if hasattr(g0[et], 'edge_attr') else None
        edge_dims[et] = int(ea.size(-1)) if ea is not None else 0

    data = {
        "graphs":      all_graphs,
        "train_size":  len(train_graphs),
        "val_size":    len(val_graphs),
        "test_size":   len(test_graphs),
        "node_dims":   node_dims,
        "edge_dims":   edge_dims,
        "edge_types":  edge_types,
        "depth_mean":  depth_mean,
        "depth_std":   depth_std,
        "feat_means":  feat_means.tolist(),
        "feat_stds":   feat_stds.tolist(),
    }

    torch.save(data, out_path)
    print(f"\n✓ Saved {len(all_graphs)} graphs to {out_path}")
    print(f"  Train: {len(train_graphs)}  Val: {len(val_graphs)}  Test: {len(test_graphs)}")
    print(f"  node_dims: {node_dims}")
    print(f"  edge_dims: {edge_dims}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main(parser.parse_args())
