"""Shared helpers for the analysis scripts.

Paths follow the layout in README.md. Run scripts from the repository root.
"""
from __future__ import annotations

import gc
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from hydrognn.data_loader import lines_as_nodes
from hydrognn.model import HydroGNN, schema_from_graph
from hydrognn.train import (add_sensor_context_features, apply_mask,
                            build_fixed_mask, compute_climatology,
                            load_sensor_indices, load_system_type_idx)

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
OUT = REPO / "outputs"
PRED = OUT / "predictions"
RESULTS = REPO / "results"

# Heusden conduit units before preprocessing/heusden/fix_units_si.py
IN_TO_M, STRICKLER, PCT_TO_FRAC = 0.0254, 25.0, 0.01

TOPOLOGY = {
    "heusden": DATA / "raw/heusden/Detailed_heusden_topology/1D rioleringsmodel Heusden!_node.csv",
    "bellinge": DATA / "interim/bellinge/topology/BellingeSWMM_node.csv",
    "tuindorp": DATA / ("raw/tuindorp/Tuindorp development - 1 min resolution/networks/"
                        "Tuindorp development - 1 min resolution_node.csv"),
}

# Held-out test evaluation at 1% coverage. Heusden has five event-isolated folds.
NETWORKS = {
    "heusden": dict(
        folds=[0, 1, 2, 3, 4],
        data=lambda k: DATA / f"processed/heusden_v3/folds/heusden_v3_fold{k}.pt",
        run=lambda k, cov=1: OUT / f"hydrognn/heusden_{cov}pct_fold{k}"),
    "bellinge": dict(
        folds=[None],
        data=lambda k: DATA / "processed/bellinge_v3_nopumpstatus.pt",
        run=lambda k, cov=1: OUT / f"hydrognn/bellinge_{cov}pct"),
    "tuindorp": dict(
        folds=[None],
        data=lambda k: DATA / "processed/tuindorp_v2_rain.pt",
        run=lambda k, cov=1: OUT / f"hydrognn/tuindorp_{cov}pct"),
}


def sensor_file(net: str, cov: int = 1) -> Path:
    return DATA / f"sensors/sensors_{net}_kmeans_{cov}pct.txt"


def fold_lengths(cov: int = 1) -> list[int]:
    """Test records per Heusden fold, in the order the folds are concatenated."""
    import json
    return [json.loads((NETWORKS["heusden"]["run"](k, cov) / "test_metrics.json").read_text())
            ["n_test_snapshots"] for k in range(5)]


def load_predictions(net: str):
    """(pred_cm, true_cm, non_sensor) written by analysis/predict_test.py."""
    d = np.load(PRED / f"{net}.npz", allow_pickle=True)
    return d["preds"], d["trues"], np.asarray(d["non_sensor"]).astype(bool)


def diagnostics(pred: np.ndarray, true: np.ndarray, nonsensor: np.ndarray) -> dict:
    """Table 2 metrics over masked manholes.

    Wet records: upper third of the test-split record-mean depth. DynRatio, rho
    and per-node NSE are medians over manholes whose test-split temporal std is
    at least 1e-3 of the std of all scored targets.
    """
    p, t = pred[:, nonsensor], true[:, nonsensor]
    err = p - t
    rec_mean = t.mean(1)
    wet = rec_mean >= np.quantile(rec_mean, 2.0 / 3.0)
    keep = t.std(0) >= 1e-3 * t.std()
    pk, tk = p[:, keep], t[:, keep]
    pc, tc = pk - pk.mean(0), tk - tk.mean(0)
    den = np.sqrt((pc ** 2).sum(0) * (tc ** 2).sum(0))
    rho = np.where(den > 1e-12, (pc * tc).sum(0) / np.maximum(den, 1e-12), 0.0)
    nse_node = 1.0 - ((pk - tk) ** 2).sum(0) / (tc ** 2).sum(0)
    return dict(mae=float(np.abs(err).mean()),
                wet_mae=float(np.abs(err[wet]).mean()),
                rmse=float(np.sqrt((err ** 2).mean())),
                dyn_ratio=float(np.median(pk.std(0) / tk.std(0))),
                rho=float(np.median(rho)),
                nse_pooled=float(1.0 - (err ** 2).sum() / ((t - t.mean()) ** 2).sum()),
                nse_node=float(np.median(nse_node)),
                n_scored=int(nonsensor.sum()), n_excluded=int((~keep).sum()))


def revert_units_si(graphs) -> int:
    """Undo the SI conduit conversion in memory. Returns conduit rows touched."""
    ets = [et for et in graphs[0].edge_types if et[1] == "conduit"]
    if not ets:
        return 0
    if abs(float(graphs[0][ets[0]].edge_attr[0, 2]) - STRICKLER) < 1e-6:
        return 0                                   # already in source units
    n = 0
    for g in graphs:
        for et in ets:
            ea = getattr(g[et], "edge_attr", None)
            if ea is None or ea.size(1) < 4:
                continue
            ea[:, 0] /= IN_TO_M
            ea[:, 2] = STRICKLER
            ea[:, 3] /= PCT_TO_FRAC
            n += ea.size(0)
    return n


def predict_test(ckpt: Path, raw, sensors_file: Path, topology_csv: Path, device):
    """Return (pred_cm, true_cm) of shape [T, N] plus the non-sensor mask."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    n_tr, n_val = int(raw["train_size"]), int(raw["val_size"])
    n_te = int(raw.get("test_size", len(raw["graphs"]) - n_tr - n_val))
    if n_te <= 0:
        raise RuntimeError("no test split")
    dmean, dstd = float(raw["depth_mean"]), float(raw["depth_std"])
    qmax = a.get("add_qmax_feature", False)
    sc = a.get("manhole_shortcut", False)
    na = a.get("normalize_assets", False)

    train_g = [lines_as_nodes(g, qmax, sc, na) for g in raw["graphs"][:n_tr]]
    test_g = [lines_as_nodes(g, qmax, sc, na)
              for g in raw["graphs"][n_tr + n_val: n_tr + n_val + n_te]]
    for graphs in (train_g, test_g):
        for g in graphs:
            for nt in g.node_types:
                x = getattr(g[nt], "x", None)
                if x is not None and torch.isnan(x).any():
                    torch.nan_to_num_(x, nan=0.0)
            for et in g.edge_types:
                ea = getattr(g[et], "edge_attr", None)
                if ea is not None and torch.isnan(ea).any():
                    torch.nan_to_num_(ea, nan=0.0)

    n = train_g[0]["manhole"].x.size(0)
    nf = train_g[0]["manhole"].x.size(1)
    sfi, didx = nf - 1, list(range(3, nf - 1))
    mask = build_fixed_mask(n, load_sensor_indices(sensors_file))
    tem = [apply_mask(g, mask, didx, sfi) for g in test_g]
    if a.get("wetness_features"):
        # Match the checkpoint's climatology scope; pre-fix checkpoints have no flag and
        # were leaky, so absent means True.
        clim_mask = None if a.get("leaky_climatology", True) else ~mask
        cmu, csd = compute_climatology([apply_mask(g, mask, didx, sfi) for g in train_g],
                                       sensor_mask=clim_mask)
        # The 4 network-wide order statistics are appended only when the run asked
        # for them; omitting them here silently changes the manhole input width.
        inc_global = a.get("global_context", False)
        tem = [add_sensor_context_features(g, cmu, csd, didx[0],
                                           include_global=inc_global) for g in tem]
    del train_g, test_g
    gc.collect()

    nd, ed, ets = schema_from_graph(tem[0])
    si = (load_system_type_idx(topology_csv, n)
          if a.get("use_system_type") else None)
    model = HydroGNN(node_dims=nd, edge_dims=ed, edge_types=ets,
                     hidden_dim=a["hidden_dim"], num_blocks=a["num_blocks"],
                     heads=a["heads"], dropout=a["dropout"],
                     use_softplus_output=a["softplus_output"],
                     use_system_type_conditioning=a.get("use_system_type", False),
                     use_input_skip=a.get("input_skip", False)).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    P, Y = [], []
    with torch.no_grad():
        for b in DataLoader(tem, batch_size=a["batch_size"], shuffle=False):
            b = b.to(device)
            ng = b["manhole"].x.size(0) // n
            sti = si.to(device).repeat(ng) if si is not None else None
            ea = {e: (b[e].edge_attr if "edge_attr" in b[e] else None)
                  for e in b.edge_types}
            pr = model(b.x_dict, b.edge_index_dict, ea, system_type_idx=sti)
            y = b["manhole"].y
            y = y.squeeze(-1) if y.ndim == 2 else y
            P.append((pr * dstd + dmean).view(ng, n).cpu().numpy() * 100.0)
            Y.append((y * dstd + dmean).view(ng, n).cpu().numpy() * 100.0)
    return np.concatenate(P), np.concatenate(Y), mask.numpy()


def per_node(pred: np.ndarray, true: np.ndarray) -> dict:
    """Per-node MAE/RMSE/NSE and std, all in cm, over the time axis."""
    err = pred - true
    mae = np.abs(err).mean(0)
    rmse = np.sqrt((err ** 2).mean(0))
    denom = ((true - true.mean(0)) ** 2).sum(0)
    nse = np.where(denom > 1e-9, 1.0 - (err ** 2).sum(0) / np.maximum(denom, 1e-9),
                   np.nan)
    return dict(mae_node_cm=mae.astype(np.float32),
                rmse_node_cm=rmse.astype(np.float32),
                nse_node=nse.astype(np.float32),
                pred_std_node_cm=pred.std(0).astype(np.float32),
                true_std_node_cm=true.std(0).astype(np.float32))


def globals_(pred: np.ndarray, true: np.ndarray, nonsensor: np.ndarray) -> dict:
    """Aggregate metrics over scored (non-sensor) nodes only, matching the text."""
    p, t = pred[:, nonsensor], true[:, nonsensor]
    err = p - t
    ss_res = float((err ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return dict(mae_global_cm=np.float32(np.abs(err).mean()),
                rmse_global_cm=np.float32(np.sqrt((err ** 2).mean())),
                r2_global=np.float32(1.0 - ss_res / max(ss_tot, 1e-9)))


def targets_cm(graphs, dmean: float, dstd: float) -> np.ndarray:
    ys = []
    for g in graphs:
        y = g["manhole"].y
        ys.append((y[:, 0] if y.dim() == 2 else y).numpy())
    return (np.stack(ys) * dstd + dmean) * 100.0


def manhole_adjacency(g, n: int) -> list[list[int]]:
    """Manhole-to-manhole adjacency induced by the hoisted asset nodes.

    In the lines-as-nodes graph a manhole reaches another only through an asset
    node, so we compose has_us/has_ds to recover direct manhole neighbours.
    """
    ln = lines_as_nodes(g, False)
    adj = [[] for _ in range(n)]
    for et in ln.edge_types:
        src, rel, dst = et
        if src != "manhole" or dst != "manhole":
            continue
        ei = ln[et].edge_index
        for u, v in zip(ei[0].tolist(), ei[1].tolist()):
            if u < n and v < n:
                adj[u].append(v)
                adj[v].append(u)
    # compose manhole -> asset -> manhole
    for asset in ("pipe", "weir", "pump"):
        us = ("manhole", "has_us", asset)
        ds = (asset, "has_ds", "manhole")
        if us not in ln.edge_types or ds not in ln.edge_types:
            continue
        a_us, a_ds = {}, {}
        ei = ln[us].edge_index
        for m, a in zip(ei[0].tolist(), ei[1].tolist()):
            a_us.setdefault(a, []).append(m)
        ei = ln[ds].edge_index
        for a, m in zip(ei[0].tolist(), ei[1].tolist()):
            a_ds.setdefault(a, []).append(m)
        for a, ups in a_us.items():
            for u in ups:
                for v in a_ds.get(a, []):
                    if u < n and v < n:
                        adj[u].append(v)
                        adj[v].append(u)
    return adj


def nearest_sensor_map(adj: list[list[int]], sensors: list[int], n: int) -> np.ndarray:
    """Multi-source BFS: index of the sensor that reaches each manhole first."""
    owner = np.full(n, -1, dtype=np.int64)
    q = deque()
    for s in sorted(sensors):
        if s < n and owner[s] == -1:
            owner[s] = s
            q.append(s)
    while q:
        u = q.popleft()
        for v in adj[u]:
            if owner[v] == -1:
                owner[v] = owner[u]
                q.append(v)
    return owner
