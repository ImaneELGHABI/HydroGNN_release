#!/usr/bin/env python3
"""Feature helpers for the Heusden fold datasets (used by build_fold_datasets.py).

Adds a pump_status column to pump edges. Pump edge attributes:
  [0] discharge_m3s
  [1] switch_on_level_m_AD
  [2] switch_off_level_m_AD
  [3] efficiency (0.8)
  [4] pump_status: upstream water level > switch-on level (0/1)
Storage and outfall nodes are kept as they are in the source graphs.
"""
from pathlib import Path
import torch
import pandas as pd
import numpy as np

REPO     = Path(__file__).resolve().parents[2]

TOPO_DIR = REPO / "data/raw/heusden/Detailed_heusden_topology"
NODE_CSV = TOPO_DIR / "1D rioleringsmodel Heusden!_node.csv"
PUMP_CSV = TOPO_DIR / "1D rioleringsmodel Heusden!_pump.csv"

KEEP_COLS   = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # drop chamber/shaft area (cols 1,2)
DEPTH_COLS  = [3, 4, 5, 6, 7, 8]
STATIC_COLS = [0, 1, 2]
FLAG_COL    = 9
RAW_DEPTH_COL = 5   # index in 12-feat raw layout (depth_current in metres above invert)


def load_topology():
    nodes    = pd.read_csv(NODE_CSV, encoding="latin1")
    manholes = nodes[nodes["node_type"] == "Manhole"].reset_index(drop=True)
    storage  = nodes[nodes["node_type"] == "Storage"].reset_index(drop=True)
    outfall  = nodes[nodes["node_type"] == "Outfall"].reset_index(drop=True)

    mh_invert  = manholes["chamber_floor"].astype(float).values
    st_invert  = storage["chamber_floor"].astype(float).values
    of_invert  = outfall["chamber_floor"].astype(float).values

    mh_id2idx  = {str(nid): i for i, nid in enumerate(manholes["node_id"])}
    st_id2idx  = {str(nid): i for i, nid in enumerate(storage["node_id"])}

    pumps = pd.read_csv(PUMP_CSV, encoding="latin1")

    print(f"  Manholes={len(manholes)}  Storage={len(storage)}  Outfall={len(outfall)}")
    print(f"  Pump edges={len(pumps)}")
    return mh_invert, st_invert, of_invert, mh_id2idx, st_id2idx, pumps


def compute_stats(graphs):
    all_x = torch.cat([g["manhole"].x[:, KEEP_COLS] for g in graphs])
    static_mean = all_x[:, STATIC_COLS].mean(0)
    static_std  = all_x[:, STATIC_COLS].std(0).clamp(min=1e-6)
    depth_mean  = float(all_x[:, 3].mean())
    depth_std   = float(all_x[:, 3].std().clamp(min=1e-6))
    return depth_mean, depth_std, static_mean, static_std


def add_pump_status(g, mh_invert, st_invert, mh_id2idx, st_id2idx, pumps):
    """Inject pump_status column into every pump edge type."""
    # Raw manhole depths [N_mh] in metres above invert
    mh_depth_raw = g["manhole"].x[:, RAW_DEPTH_COL]   # already raw before normalise_graph
    # Storage nodes: feature layout in source is [ground_level, x, y, capacity?]
    # depth is NOT in the storage feature vector directly — use zero as fallback
    st_depth_raw = None
    if hasattr(g["storage"], "x") and g["storage"].x is not None:
        # storage x has 4 features: we don't have explicit depth there
        # best proxy: 0 (unknown) — pump_status will default to switch-off comparison
        st_depth_raw = torch.zeros(g["storage"].x.shape[0])

    pump_types = [et for et in g.edge_types if et[1] == "pump"]
    for et in pump_types:
        s_type, _, d_type = et
        ei  = g[et].edge_index   # [2, E]
        ea  = g[et].edge_attr    # [E, 4]

        # Build pump_status column [E]
        status = torch.zeros(ea.shape[0])

        # Find which pump_df rows correspond to this edge type
        src_col = "us_node_id"
        dst_col = "ds_node_id"
        edge_row = 0
        for _, row in pumps.iterrows():
            us_id      = str(row.get(src_col, ""))
            switch_on  = float(row.get("switch_on_level",  0) or 0)
            switch_off = float(row.get("switch_off_level", 0) or 0)

            # Determine node type of upstream node
            if us_id in mh_id2idx:
                us_nt  = "manhole"
                us_idx = mh_id2idx[us_id]
                us_inv = float(mh_invert[us_idx])
                us_dep = float(mh_depth_raw[us_idx])
                us_wl  = us_dep + us_inv    # water surface m AD
            elif us_id in st_id2idx:
                us_nt  = "storage"
                us_idx = st_id2idx[us_id]
                us_inv = float(st_invert[us_idx])
                us_dep = 0.0                # not directly available
                us_wl  = us_inv             # conservative: at invert
            else:
                continue                    # unknown upstream node

            # Check if this pump row belongs to the current (s_type, d_type)
            if us_nt != s_type:
                continue

            if edge_row >= ea.shape[0]:
                break

            # Pump status: on if upstream water surface > switch_on_level (m AD for Heusden)
            pump_on = 1.0 if us_wl > switch_on else 0.0
            status[edge_row] = pump_on
            edge_row += 1

        # Append as 5th column
        g[et].edge_attr = torch.cat([ea, status.unsqueeze(1)], dim=1)

    return g


def normalise_graph(g, depth_mean, depth_std, static_mean, static_std, mh_invert_t):
    g = g.clone()
    x = g["manhole"].x[:, KEEP_COLS].clone()
    x[:, STATIC_COLS]  = (x[:, STATIC_COLS] - static_mean.to(x.device)) / static_std.to(x.device)
    x[:, DEPTH_COLS]   = (x[:, DEPTH_COLS]  - depth_mean) / depth_std
    x[:, FLAG_COL]     = 1.0
    g["manhole"].x     = x
    g["manhole"].y     = (g["manhole"].y - depth_mean) / depth_std
    g["manhole"].invert_level_m = mh_invert_t
    return g
