"""Append per-node inflow to the Tuindorp dataset.

Snapshots carry no provenance, so each is matched to its simulation and row
by comparing hydraulic_head - elevation with the de-normalised target. The
matching total_inflow.csv row is then attached.

Output: data/processed/tuindorp_v2_rain.pt
  manhole.x gains 3 columns (log1p, z-normalised with train statistics):
    inflow_t      total inflow at t          [m3/s]
    inflow_mean5  mean over rows t-4..t      [m3/s]
    inflow_max5   max over rows t-4..t       [m3/s]

    python preprocessing/tuindorp/attach_inflow.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
CACHE_IN  = REPO / "data/processed/tuindorp_v2.pt"
CACHE_OUT = REPO / "data/processed/tuindorp_v2_rain.pt"
TOPO_CSV  = REPO / ("data/raw/tuindorp/Tuindorp development - "
                    "1 min resolution/networks/Tuindorp development - 1 min resolution_node.csv")
SIM_ROOT  = REPO / "data/raw/tuindorp/Tuindorp development - 1 min resolution/simulations"
SPLITS = ["training", "validation", "testing"]
N_FP = 32
WINDOW = 5


def manhole_order():
    df = pd.read_csv(TOPO_CSV, encoding="latin-1")
    mh = df[df["node_type"] == "Junction"].reset_index(drop=True)
    ids = mh["node_id"].astype(str).str.strip().tolist()   # "j_9006F", ...
    elev = mh["elevation"].astype(float).values
    return ids, elev


def load_node_csv(path: Path, node_ids: list, suffix: str) -> np.ndarray:
    """Return [T, N_manholes] matrix; columns are 'node_<id>_<suffix>'."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {c: c for c in df.columns}
    T = len(df)
    out = np.full((T, len(node_ids)), np.nan, dtype=np.float64)
    for j, nid in enumerate(node_ids):
        col = f"node_{nid}_{suffix}"
        if col in col_map:
            out[:, j] = pd.to_numeric(df[col], errors="coerce").values
    return out


def main():
    print(f"Loading cache: {CACHE_IN}")
    raw = torch.load(CACHE_IN, map_location="cpu", weights_only=False)
    all_graphs = raw["graphs"]
    n_tr, n_val = int(raw["train_size"]), int(raw["val_size"])
    N = all_graphs[0]["manhole"].x.size(0)
    depth_mean, depth_std = float(raw["depth_mean"]), float(raw["depth_std"])
    print(f"  snapshots: {len(all_graphs)} (train {n_tr}, val {n_val})  manholes: {N}")

    node_ids, elev = manhole_order()
    assert len(node_ids) == N, f"topology manholes {len(node_ids)} != cache {N}"

    Y = torch.stack([g["manhole"].y[:, 0] if g["manhole"].y.dim() == 2
                     else g["manhole"].y for g in all_graphs]).numpy()
    Y_phys = Y * depth_std + depth_mean          # physical depth above invert
    fp_nodes = np.argsort(-Y.std(0))[:N_FP]

    # ── enumerate all real_N sim dirs ────────────────────────────────────────
    sim_dirs = []
    for split in SPLITS:
        d = SIM_ROOT / split
        if d.exists():
            sim_dirs.extend(sorted(d.iterdir()))
    print(f"  sim dirs: {len(sim_dirs)}")

    cand_keys, cand_rows = [], []
    sim_data = {}
    for si, sd in enumerate(sim_dirs):
        hh_path = sd / "hydraulic_head.csv"
        if not hh_path.exists():
            continue
        HH = load_node_csv(hh_path, node_ids, "Hydraulic_head")
        depth_phys = HH - elev[None, :]          # -> physical depth above invert
        sim_data[si] = sd
        for t in range(depth_phys.shape[0]):
            cand_keys.append((si, t))
            cand_rows.append(depth_phys[t])
    cand = np.stack(cand_rows)
    print(f"  candidate rows: {cand.shape[0]}")

    cand_fp = cand[:, fp_nodes]
    d0 = np.nanmax(np.abs(cand_fp - Y_phys[0][fp_nodes]), axis=1)
    print(f"  fingerprint check (snapshot 0): min diff = {np.nanmin(d0):.2e}")
    if np.nanmin(d0) > 1e-3:
        sys.exit("ERROR: no candidate row matches snapshot 0 — alignment impossible.")

    matches, worst = [], 0.0
    for i in range(len(all_graphs)):
        d = np.nanmax(np.abs(cand_fp - Y_phys[i][fp_nodes]), axis=1)
        j = int(np.nanargmin(d))
        if d[j] > 1e-3:
            sys.exit(f"ERROR: snapshot {i} unmatched (best diff {d[j]:.4f}) — aborting.")
        worst = max(worst, float(d[j]))
        matches.append(cand_keys[j])
    dup = len(matches) - len(set(matches))
    print(f"  matched {len(matches)}/{len(all_graphs)}  worst diff {worst:.2e}  duplicates: {dup}")

    # ── attach total_inflow ──────────────────────────────────────────────────
    inflow_cache = {}
    feats = np.zeros((len(all_graphs), N, 3), dtype=np.float32)
    n_missing = 0
    for i, (si, t) in enumerate(matches):
        sd = sim_data[si]
        if si not in inflow_cache:
            ip = sd / "total_inflow.csv"
            inflow_cache[si] = load_node_csv(ip, node_ids, "Total_inflow") if ip.exists() else None
            if inflow_cache[si] is None:
                n_missing += 1
        Q = inflow_cache[si]
        if Q is None:
            continue
        lo = max(0, t - (WINDOW - 1))
        win = np.nan_to_num(Q[lo:t + 1], nan=0.0)
        feats[i, :, 0] = np.nan_to_num(Q[t], nan=0.0)
        feats[i, :, 1] = win.mean(0)
        feats[i, :, 2] = win.max(0)
    if n_missing:
        print(f"  [warn] {n_missing} sims had no total_inflow.csv")

    # ── log1p + z-normalise with train stats ─────────────────────────────────
    feats = np.log1p(np.clip(feats, 0.0, None))
    tr = feats[:n_tr].reshape(-1, 3)
    mu, sd_ = tr.mean(0), np.clip(tr.std(0), 1e-9, None)
    feats = (feats - mu) / sd_
    print(f"  inflow feature stats (log1p train): mu={mu}  sd={sd_}")

    for i, g in enumerate(all_graphs):
        g["manhole"].x = torch.cat(
            [g["manhole"].x, torch.from_numpy(feats[i]).float()], dim=1)
    out = dict(raw)
    out["graphs"] = all_graphs
    out["inflow_norm"] = {"mu": mu.tolist(), "sd": sd_.tolist()}
    out["provenance"] = [(sim_data[si].name, t) for si, t in matches]
    torch.save(out, CACHE_OUT)
    print(f"Saved: {CACHE_OUT}  (manhole dim {all_graphs[0]['manhole'].x.size(1)})")


if __name__ == "__main__":
    main()
