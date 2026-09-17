#!/usr/bin/env python3
"""Build the Bellinge dataset from the simulated depnod CSVs.

Storage units and outfalls are included as always-observed context nodes
(sensor_flag = 1, never masked, never scored).

Node features (10):
  [0] ground_level  (z-norm)
  [1] x             (z-norm)
  [2] y             (z-norm)
  [3] depth_current (z-norm)
  [4] temp_recent   (z-norm)
  [5] temp_mean     (z-norm)
  [6] temp_max      (z-norm)
  [7] temp_velocity (z-norm)
  [8] temp_trend    (z-norm)
  [9] sensor_flag

Pump edge attributes (5; column 4 is removed by strip_pump_status.py):
  [0] discharge_m3s
  [1] switch_on_level_m
  [2] switch_off_level_m
  [3] efficiency (0.8 placeholder)
  [4] pump_status (0/1, per timestep)
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

REPO      = Path(__file__).resolve().parents[2]
SIM_DIR   = REPO / "data/interim/bellinge/simulations"
EVENT_CSV = REPO / "data/interim/bellinge/event_list.csv"
TOPO_DIR  = REPO / "data/interim/bellinge/topology"
OUT_DIR   = REPO / "data/processed"
OUT_PATH  = OUT_DIR / "bellinge_v3.pt"

W                = 12   # temporal window
WARMUP           = 12
SAMPLES_PER_SIM  = 15
SEED             = 42

# CSV node_type values → graph node type names (lowercase, matching Heusden/Tuindorp)
CSV_TO_NT = {"Manhole": "manhole", "StorageUnit": "storage", "Outfall": "outfall"}
NODE_TYPES = ["manhole", "storage", "outfall"]  # order determines graph construction


# ── topology ─────────────────────────────────────────────────────────────────

def load_topology():
    node_csv = next(TOPO_DIR.glob("*_node.csv"))
    nodes    = pd.read_csv(node_csv, encoding="latin1")

    csv_types = list(CSV_TO_NT.keys())
    all_nodes = nodes[nodes["node_type"].isin(csv_types)].reset_index(drop=True)
    # Rename node_type column to use graph names
    all_nodes = all_nodes.copy()
    all_nodes["node_type"] = all_nodes["node_type"].map(CSV_TO_NT)

    def _load(pat):
        f = next(TOPO_DIR.glob(pat), None)
        return pd.read_csv(f, encoding="latin1") if f else pd.DataFrame()

    edge_tables = {t: _load(f"*_{t}.csv")
                   for t in ["conduit", "weir", "pump", "orifice"]}
    return all_nodes, edge_tables


def build_all_indices(all_nodes: pd.DataFrame):
    """Return per-type ordered id lists and a global id→(type, local_idx) map."""
    per_type: Dict[str, pd.DataFrame] = {}
    for nt in NODE_TYPES:
        df = all_nodes[all_nodes["node_type"] == nt].reset_index(drop=True)
        per_type[nt] = df

    # global id → (node_type, local_idx within that type)
    global_map: Dict[str, tuple] = {}
    for nt, df in per_type.items():
        for i, nid in enumerate(df["node_id"]):
            global_map[nid] = (nt, i)

    return per_type, global_map


def build_edges(edge_tables: dict, global_map: dict, pump_df: pd.DataFrame):
    """Return edges_by_type with pump having 5-dim placeholder attr (status=0).

    For non-pump edges: [diam, length/100, roughness, slope]
    For pump edges:     [discharge, switch_on, switch_off, efficiency, pump_status]
    pump_status is filled per-timestep in graphs_from_simulation.
    """
    edges_by_type: Dict[tuple, tuple] = {}

    for asset_type, df in edge_tables.items():
        if df.empty:
            continue
        src_col = "from_node" if "from_node" in df.columns else (
                  "us_node_id" if "us_node_id" in df.columns else None)
        dst_col = "to_node" if "to_node" in df.columns else (
                  "ds_node_id" if "ds_node_id" in df.columns else None)
        if src_col is None or dst_col is None:
            continue

        # Group by (src_type, dst_type) since HeteroData needs separate tensors
        buckets: Dict[tuple, tuple] = {}  # (src_type, dst_type) -> (srcs, dsts, attrs)

        for _, row in df.iterrows():
            s_id = str(row.get(src_col, ""))
            d_id = str(row.get(dst_col, ""))
            if s_id not in global_map or d_id not in global_map:
                continue
            s_type, s_idx = global_map[s_id]
            d_type, d_idx = global_map[d_id]

            if asset_type == "conduit" or asset_type == "orifice":
                diam   = float(row.get("conduit_width",  row.get("diameter", 0)) or 0)
                length = float(row.get("conduit_length", row.get("length",   0)) or 0)
                rough  = float(row.get("bottom_roughness_Manning",
                                       row.get("roughness", 0)) or 0)
                slope  = float(row.get("gradient", 0) or 0)
                attr = [diam, length / 100.0, rough, slope]
            elif asset_type == "weir":
                diam  = float(row.get("width", 0) or 0)
                attr  = [diam, 0.0, float(row.get("discharge_coeff", 0) or 0), 0.0]
            elif asset_type == "pump":
                discharge  = float(row.get("discharge",        0) or 0)
                switch_on  = float(row.get("switch_on_level",  0) or 0)
                switch_off = float(row.get("switch_off_level", 0) or 0)
                attr = [discharge, switch_on, switch_off, 0.8, 0.0]  # status=0 placeholder
            else:
                attr = [0.0, 0.0, 0.0, 0.0]

            key = (s_type, asset_type, d_type)
            if key not in buckets:
                buckets[key] = ([], [], [])
            buckets[key][0].append(s_idx)
            buckets[key][1].append(d_idx)
            buckets[key][2].append(attr)

        for (s_type, a_type, d_type), (srcs, dsts, attrs) in buckets.items():
            et = (s_type, a_type, d_type)
            edges_by_type[et] = (
                torch.tensor([srcs, dsts], dtype=torch.long),
                torch.tensor(attrs, dtype=torch.float32),
            )

    return edges_by_type


# ── temporal features ─────────────────────────────────────────────────────────

def temporal_features(depths: np.ndarray, t: int, w: int) -> np.ndarray:
    """depths: [T, N] → [N, 6] features at step t."""
    window  = depths[max(0, t - w): t + 1]
    depth_t = depths[t]
    recent  = depths[t - 1] if t > 0 else depth_t
    mean_d  = window.mean(axis=0)
    max_d   = window.max(axis=0)
    velocity = (depth_t - depths[max(0, t - w)]) / max(1, min(t, w))
    L = len(window)
    if L > 1:
        xs = np.arange(L, dtype=float); xs -= xs.mean()
        denom = (xs ** 2).sum()
        trend = ((xs[:, None] * (window - window.mean(axis=0))).sum(axis=0) / denom
                 if denom > 0 else np.zeros(depths.shape[1]))
    else:
        trend = np.zeros(depths.shape[1])
    return np.stack([depth_t, recent, mean_d, max_d, velocity, trend], axis=1)


# ── graph builder ─────────────────────────────────────────────────────────────

def make_graph(
    node_feats_by_type: Dict[str, np.ndarray],
    manhole_targets: np.ndarray,
    invert_by_type: Dict[str, np.ndarray],
    edges_by_type: dict,
) -> HeteroData:
    g = HeteroData()
    for nt, feats in node_feats_by_type.items():
        g[nt].x = torch.tensor(feats, dtype=torch.float32)
        if nt == "manhole":
            g[nt].y = torch.tensor(manhole_targets, dtype=torch.float32)
        g[nt].invert_level_m = torch.tensor(invert_by_type[nt], dtype=torch.float32)
    for et, (ei, ea) in edges_by_type.items():
        g[et].edge_index = ei
        g[et].edge_attr  = ea
    return g


# ── per-simulation builder ────────────────────────────────────────────────────

def graphs_from_simulation(
    depnod_path: Path,
    per_type: Dict[str, pd.DataFrame],
    edges_static: dict,      # pump attr col 4 is placeholder 0
    pump_df: pd.DataFrame,   # raw pump table for status computation
    n_samples: int,
    rng: np.random.Generator,
) -> List[HeteroData]:

    try:
        df = pd.read_csv(depnod_path, encoding="latin1", header=0, skiprows=[1])
    except Exception as e:
        print(f"    WARNING: {depnod_path.name}: {e}")
        return []

    # Build depth matrices per node type [T, N_type]
    depth_by_type: Dict[str, np.ndarray] = {}
    invert_by_type: Dict[str, np.ndarray] = {}
    ground_by_type: Dict[str, np.ndarray] = {}
    x_by_type:      Dict[str, np.ndarray] = {}
    y_by_type:      Dict[str, np.ndarray] = {}
    wl_by_type:     Dict[str, np.ndarray] = {}  # raw water surface m AD [T, N]

    T = None
    for nt, df_nt in per_type.items():
        node_ids = list(df_nt["node_id"])
        invert_m = df_nt["chamber_floor"].astype(float).values
        ground_m = df_nt["ground_level"].astype(float).values
        x_coord  = df_nt["x"].astype(float).values
        y_coord  = df_nt["y"].astype(float).values

        present  = [nid for nid in node_ids if nid in df.columns]
        if len(present) < max(1, len(node_ids) * 0.3):
            # very few nodes present — pad with zeros
            print(f"    WARNING: {nt} has only {len(present)}/{len(node_ids)} nodes in {depnod_path.name}")

        # Water surface levels [T, N_present]
        wl = df[present].values.astype(np.float32) if present else np.zeros((len(df), 0), np.float32)
        if T is None:
            T = len(wl)

        # Depth above invert per present node
        inv_present = np.array([invert_m[node_ids.index(nid)] for nid in present])
        depth_present = np.maximum(wl - inv_present[None, :], 0.0)

        # Full [T, N] matrix with zeros for missing nodes
        N = len(node_ids)
        depth_full = np.zeros((T, N), dtype=np.float32)
        wl_full    = np.zeros((T, N), dtype=np.float32)
        present_idx = [node_ids.index(nid) for nid in present]
        if present_idx:
            depth_full[:, present_idx] = depth_present
            wl_full[:, present_idx]    = wl

        depth_by_type[nt]  = depth_full
        wl_by_type[nt]     = wl_full
        invert_by_type[nt] = invert_m
        ground_by_type[nt] = ground_m
        x_by_type[nt]      = x_coord
        y_by_type[nt]      = y_coord

    if T is None or T == 0:
        return []

    # Build pump status lookup: pump_edge_key → array [T] of 0/1
    # switch_on_level is depth above invert (Bellinge convention)
    manhole_ids  = list(per_type["manhole"]["node_id"])
    storage_ids  = list(per_type["storage"]["node_id"])

    def _depth_series(node_id: str) -> Optional[np.ndarray]:
        if node_id in [per_type["Manhole"]["node_id"].tolist()]:
            idx = manhole_ids.index(node_id)
            return depth_by_type["Manhole"][:, idx]
        if node_id in storage_ids:
            idx = storage_ids.index(node_id)
            return depth_by_type["StorageUnit"][:, idx]
        return None

    # Build pump_status_by_edge: (us_id, ds_id, link_suffix?) → [T]
    pump_status_map = {}  # row_index_in_pump_df → [T]
    if not pump_df.empty:
        src_col = "us_node_id" if "us_node_id" in pump_df.columns else "from_node"
        dst_col = "ds_node_id" if "ds_node_id" in pump_df.columns else "to_node"
        for ridx, row in pump_df.iterrows():
            us_id      = str(row.get(src_col, ""))
            switch_on  = float(row.get("switch_on_level", 0) or 0)
            switch_off = float(row.get("switch_off_level", 0) or 0)
            # Determine depth series at upstream node
            depth_us = None
            for nt in ["manhole", "storage"]:
                ids = list(per_type[nt]["node_id"])
                if us_id in ids:
                    idx = ids.index(us_id)
                    depth_us = depth_by_type[nt][:, idx]
                    break
            if depth_us is None:
                pump_status_map[ridx] = np.zeros(T, dtype=np.float32)
                continue
            # Hysteresis: on when depth > switch_on; stays on until depth < switch_off
            status = np.zeros(T, dtype=np.float32)
            on = False
            for ti in range(T):
                d = depth_us[ti]
                if not on and d >= switch_on:
                    on = True
                elif on and d <= switch_off:
                    on = False
                status[ti] = float(on)
            pump_status_map[ridx] = status

    # Sample timesteps
    eligible = list(range(W + WARMUP, T))
    if not eligible:
        return []
    chosen = sorted(rng.choice(eligible, size=min(n_samples, len(eligible)), replace=False))

    graphs = []
    for t in chosen:
        # Build per-type feature matrices
        node_feats_by_type = {}
        for nt in NODE_TYPES:
            N_nt = len(per_type[nt])
            tf = temporal_features(depth_by_type[nt], t, W)  # [N, 6]
            feats = np.column_stack([
                ground_by_type[nt], x_by_type[nt], y_by_type[nt],
                tf,                   # cols 3-8
                np.ones(N_nt),        # sensor_flag=1 (overwritten for manholes during training)
            ])
            node_feats_by_type[nt] = feats

        manhole_targets = depth_by_type["manhole"][t]

        # Inject pump status at timestep t into edge_attr
        edges_t = {}
        for et, (ei, ea) in edges_static.items():
            if et[1] == "pump" and not pump_df.empty:
                # Build updated pump edge_attr with correct status at t
                ea_new = ea.clone()
                src_col = "us_node_id" if "us_node_id" in pump_df.columns else "from_node"
                dst_col = "ds_node_id" if "ds_node_id" in pump_df.columns else "to_node"

                # Match pump_df rows to edge_index rows by (src_type, dst_type)
                s_type, _, d_type = et
                # We need to know which pump_df row corresponds to which edge row
                # Re-derive: filter pump_df to this (s_type, d_type) combination
                global_map_local = {}
                for nt2, df_nt2 in per_type.items():
                    for i, nid in enumerate(df_nt2["node_id"]):
                        global_map_local[nid] = (nt2, i)

                row_map = []  # edge_row -> pump_df_ridx
                for ridx, row in pump_df.iterrows():
                    us = str(row.get(src_col, ""))
                    ds = str(row.get(dst_col, ""))
                    if us in global_map_local and ds in global_map_local:
                        ut, ui = global_map_local[us]
                        dt, di = global_map_local[ds]
                        if ut == s_type and dt == d_type:
                            row_map.append((ridx, ui, di))

                # Fill in pump_status for each edge row
                for edge_row, (ridx, ui, di) in enumerate(row_map):
                    if edge_row < ea_new.shape[0] and ridx in pump_status_map:
                        ea_new[edge_row, 4] = float(pump_status_map[ridx][t])

                edges_t[et] = (ei, ea_new)
            else:
                edges_t[et] = (ei, ea)

        invert_by_type_t = {nt: per_type[nt]["chamber_floor"].astype(float).values
                            for nt in NODE_TYPES}
        graphs.append(make_graph(node_feats_by_type, manhole_targets,
                                 invert_by_type_t, edges_t))
    return graphs


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Bellinge v3 preprocessing — storage/outfall nodes + pump status")
    print("=" * 70)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\nLoading topology …")
    all_nodes, edge_tables = load_topology()
    per_type, global_map   = build_all_indices(all_nodes)
    for nt in NODE_TYPES:
        print(f"  {nt}: {len(per_type[nt])} nodes")

    pump_df = edge_tables.get("pump", pd.DataFrame())
    edges_static = build_edges(edge_tables, global_map, pump_df)
    print(f"  Edge types: {list(edges_static.keys())}")

    print("\nLoading event list …")
    with open(EVENT_CSV) as f:
        events = list(csv.DictReader(f))
    print(f"  {len(events)} events — train/val/test splits")

    rng = np.random.default_rng(SEED)
    split_graphs: dict = {"train": [], "val": [], "test": []}

    for i, ev in enumerate(events):
        tag   = ev["tag"]
        split = ev["split"]
        path  = SIM_DIR / f"Node_BellingeSWMM_{tag}_depnod.csv"
        if not path.exists():
            continue
        gs = graphs_from_simulation(path, per_type, edges_static, pump_df,
                                    SAMPLES_PER_SIM, rng)
        split_graphs[split].extend(gs)
        if (i + 1) % 100 == 0 or i == len(events) - 1:
            print(f"  [{i+1}/{len(events)}]  "
                  f"train={len(split_graphs['train'])}  "
                  f"val={len(split_graphs['val'])}  "
                  f"test={len(split_graphs['test'])}")

    train_graphs = split_graphs["train"]
    val_graphs   = split_graphs["val"]
    test_graphs  = split_graphs["test"]
    all_graphs   = train_graphs + val_graphs + test_graphs

    if not all_graphs:
        print("ERROR: no graphs built.")
        return

    print(f"\nBuilt {len(all_graphs)} total: "
          f"train={len(train_graphs)}  val={len(val_graphs)}  test={len(test_graphs)}")

    # ── normalisation from training graphs ──────────────────────────────────
    print("\nComputing normalisation stats from training graphs …")
    # depth stats: use all prediction targets (manhole depth) across training
    train_y = torch.cat([g["manhole"].y for g in train_graphs])
    depth_mean = float(train_y.mean())
    depth_std  = float(train_y.std().clamp(min=1e-6))

    # static spatial stats per node type (cols 0-2)
    static_stats: Dict[str, tuple] = {}
    for nt in NODE_TYPES:
        xs = torch.cat([g[nt].x[:, :3] for g in train_graphs])
        static_stats[nt] = (xs.mean(0), xs.std(0).clamp(min=1e-6))

    print(f"  depth_mean={depth_mean:.4f}  depth_std={depth_std:.4f}")

    DEPTH_COLS = [3, 4, 5, 6, 7, 8]
    print("Normalising …")
    for g in all_graphs:
        for nt in NODE_TYPES:
            x = g[nt].x.clone()
            sm, ss = static_stats[nt]
            x[:, :3]        = (x[:, :3] - sm) / ss
            x[:, DEPTH_COLS] = (x[:, DEPTH_COLS] - depth_mean) / depth_std
            g[nt].x = x
        g["manhole"].y = (g["manhole"].y - depth_mean) / depth_std

    # per-column z-norm for velocity (col7) and trend (col8) using manhole train stats
    for col in [7, 8]:
        vals = torch.cat([g["manhole"].x[:, col] for g in train_graphs])
        cm, cs = float(vals.mean()), float(vals.std().clamp(min=1e-6))
        for g in all_graphs:
            for nt in NODE_TYPES:
                x = g[nt].x.clone()
                x[:, col] = (x[:, col] - cm) / cs
                g[nt].x = x

    # ── package and save ────────────────────────────────────────────────────
    g0 = all_graphs[0]
    node_dims  = {nt: g0[nt].x.size(-1) for nt in g0.node_types if hasattr(g0[nt], "x")}
    edge_dims  = {et: (g0[et].edge_attr.size(-1)
                       if hasattr(g0[et], "edge_attr") and g0[et].edge_attr is not None else 0)
                  for et in g0.edge_types}
    edge_types = list(g0.edge_types)

    data = {
        "graphs":     all_graphs,
        "train_size": len(train_graphs),
        "val_size":   len(val_graphs),
        "test_size":  len(test_graphs),
        "node_dims":  node_dims,
        "edge_dims":  edge_dims,
        "edge_types": edge_types,
        "depth_mean": depth_mean,
        "depth_std":  depth_std,
    }
    torch.save(data, OUT_PATH)
    print(f"\n✓ Saved {len(all_graphs)} graphs → {OUT_PATH}")
    print(f"  node_dims: {node_dims}")
    print(f"  edge_types: {edge_types}")
    print(f"  edge_dims: {edge_dims}")


if __name__ == "__main__":
    main()
