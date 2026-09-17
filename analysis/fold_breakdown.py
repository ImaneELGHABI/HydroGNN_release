"""Per-fold breakdown of Heusden test performance.

The paper uses fold_comparison.frac_spread_explained_by_depth (Section 3.1.2)
and the per-fold storm assignment. The script also reports:

  1. MAE / RMSE and two forms of NSE: nse_pooled (variance over all scored
     elements, dominated by elevation differences between manholes) and
     nse_node (per manhole, against its own mean, then the median).
  2. Per-fold storm statistics and a pooled curve of |error| against
     network-mean depth; fold MAE is compared with its position on that curve.
  3. Flood-conditional error, with flooding defined as level >= ground level
     from the topology CSV, and POD / FAR / CSI.
  4. Error at manholes adjacent to weirs, pumps and neither, also divided by
     each group's target standard deviation.

Predictions go through the same transforms as train.score_test_split, with
the context features computed from the transformed train graphs.

    python -m analysis.fold_breakdown --runs "outputs/hydrognn/heusden_1pct_fold{k}" --tag heusden_1pct
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def _resolve(saved: str, subdir: str) -> Path:
    """Path saved in a checkpoint, or the same file under the release layout."""
    p = Path(saved)
    for cand in (p, REPO / p):
        if cand.exists():
            return cand
    return REPO / subdir / p.name

from hydrognn.data_loader import lines_as_nodes, asset_endpoint_indices   # noqa: E402
from hydrognn.datum import enforce_datum, patch_invert_units             # noqa: E402
from hydrognn.model import HydroGNN, schema_from_graph                    # noqa: E402
from hydrognn.train import (add_sensor_context_features, apply_mask,      # noqa: E402
                            build_fixed_mask, compute_climatology,
                            load_sensor_indices, load_system_type_idx)

FEET_PER_METRE = 3.280839895013123


# ── prediction ───────────────────────────────────────────────────────────────

def predict_test(run_dir: Path, device):
    """Rebuild the test predictions for one run, mirroring train.score_test_split.

    Returns (pred_m, true_m, nonsensor, meta) with pred/true in metres in the
    model's own target frame (level above datum, or depth above invert if the run
    used --datum_mode depth).
    """
    ck = torch.load(run_dir / "best_model.pth", map_location="cpu", weights_only=False)
    a = ck["args"]
    data_file = _resolve(a["data_file"], "data/processed/heusden_v3/folds")
    raw = torch.load(data_file, map_location="cpu", weights_only=False)

    n_tr, n_val = int(raw["train_size"]), int(raw["val_size"])
    n_te = int(raw.get("test_size", len(raw["graphs"]) - n_tr - n_val))
    if n_te <= 0:
        raise RuntimeError(f"{run_dir}: dataset has no test split")
    dmean, dstd = float(raw["depth_mean"]), float(raw["depth_std"])

    qmax = a.get("add_qmax_feature", False)
    sc = a.get("manhole_shortcut", False)
    na = a.get("normalize_assets", False)

    def build(gs):
        out = [lines_as_nodes(g, qmax, sc, na) for g in gs]
        for g in out:
            for nt in g.node_types:
                x = getattr(g[nt], "x", None)
                if x is not None and torch.isnan(x).any():
                    torch.nan_to_num_(x, nan=0.0)
            for et in g.edge_types:
                ea = getattr(g[et], "edge_attr", None)
                if ea is not None and torch.isnan(ea).any():
                    torch.nan_to_num_(ea, nan=0.0)
        return out

    train_g = build(raw["graphs"][:n_tr])
    test_g = build(raw["graphs"][n_tr + n_val: n_tr + n_val + n_te])

    # Same datum handling the trainer applied, on BOTH splits. The climatology
    # below is computed from the train graphs, so they must be in the same frame.
    if a.get("datum_patch_units", False):
        patch_invert_units([train_g, test_g])
    if a.get("datum_mode", "off") not in ("off", "check"):
        enforce_datum(train_g, [], dmean, dstd, mode=a["datum_mode"], verbose=False)
        enforce_datum(test_g, [], dmean, dstd, mode=a["datum_mode"], verbose=False)

    n = train_g[0]["manhole"].x.size(0)
    nf = train_g[0]["manhole"].x.size(1)
    sensor_flag_idx, depth_idxs = nf - 1, list(range(3, nf - 1))

    sensor_file = _resolve(a["sensor_file"], "data/sensors")
    mask = build_fixed_mask(n, load_sensor_indices(sensor_file))
    masked_depth_idxs = depth_idxs[:1] if a.get("mask_current_only") else depth_idxs
    if a.get("mask_depth_cols") is not None and a["mask_depth_cols"] >= 1:
        masked_depth_idxs = depth_idxs[:a["mask_depth_cols"]]

    test_masked = [apply_mask(g, mask, masked_depth_idxs, sensor_flag_idx) for g in test_g]
    if a.get("wetness_features"):
        # Match the climatology scope the checkpoint was trained under. Checkpoints
        # predating the fix carry no flag and were leaky, so absent means True; scoring
        # a leak-free model with all-node climatology would feed it a channel it never
        # saw in training.
        clim_mask = None if a.get("leaky_climatology", True) else ~mask
        cmu, csd = compute_climatology(
            [apply_mask(g, mask, masked_depth_idxs, sensor_flag_idx) for g in train_g],
            sensor_mask=clim_mask)
        test_masked = [add_sensor_context_features(
            g, cmu, csd, depth_idxs[0], include_global=a.get("global_context", False))
            for g in test_masked]

    invert_ft = train_g[0]["manhole"].invert_level_m.view(-1).numpy().copy()
    del train_g, test_g
    gc.collect()

    nd, ed, ets = schema_from_graph(test_masked[0])
    si = (load_system_type_idx(_resolve(a["topology_csv"], "data/raw/heusden/Detailed_heusden_topology"), n)
          if a.get("use_system_type") else None)
    model = HydroGNN(node_dims=nd, edge_dims=ed, edge_types=ets,
                     hidden_dim=a["hidden_dim"], num_blocks=a["num_blocks"],
                     heads=a["heads"], dropout=a["dropout"],
                     use_softplus_output=a["softplus_output"],
                     use_system_type_conditioning=a.get("use_system_type", False),
                     use_input_skip=a.get("input_skip", False),
                     use_static_highway=a.get("static_highway", False)).to(device)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()

    P, Y = [], []
    with torch.no_grad():
        for b in DataLoader(test_masked, batch_size=a["batch_size"], shuffle=False):
            b = b.to(device)
            ng = b["manhole"].x.size(0) // n
            sti = si.to(device).repeat(ng) if si is not None else None
            ea = {e: (b[e].edge_attr if "edge_attr" in b[e] else None) for e in b.edge_types}
            pr = model(b.x_dict, b.edge_index_dict, ea, system_type_idx=sti)
            y = b["manhole"].y
            y = y.squeeze(-1) if y.ndim == 2 else y
            P.append((pr * dstd + dmean).view(ng, n).cpu().numpy())
            Y.append((y * dstd + dmean).view(ng, n).cpu().numpy())

    ep = asset_endpoint_indices(test_masked[0])
    endpoints = {g: (u.numpy().copy(), d.numpy().copy()) for g, (u, d) in ep.items()}
    del test_masked, model
    gc.collect()

    prov = raw.get("event_provenance")
    test_events = ([p["bui_id"] for p in prov[n_tr + n_val: n_tr + n_val + n_te]]
                   if prov else None)
    meta = dict(args=a, ckpt_epoch=int(ck.get("epoch", -1)),
                val_mae_cm=float(ck.get("val_mae_cm", float("nan"))),
                depth_mean=dmean, depth_std=dstd, n_nodes=n,
                invert_ft=invert_ft, endpoints=endpoints,
                test_events=test_events,
                fold_assignment=raw.get("fold_assignment"),
                topology_csv=a.get("topology_csv"))
    del raw
    gc.collect()
    return np.concatenate(P), np.concatenate(Y), mask.numpy(), meta


# ── metrics ──────────────────────────────────────────────────────────────────

def elem_metrics(pred, true, sel):
    """MAE / RMSE / bias / pooled NSE over the selected [T, N] elements, in cm."""
    if sel.sum() == 0:
        return None
    p, t = pred[sel] * 100.0, true[sel] * 100.0
    e = p - t
    var = ((t - t.mean()) ** 2).sum()
    return dict(n_elements=int(sel.sum()),
                mae_cm=float(np.abs(e).mean()),
                rmse_cm=float(np.sqrt((e ** 2).mean())),
                bias_cm=float(e.mean()),
                sigma_cm=float(t.std()),
                nse_pooled=float(1.0 - (e ** 2).sum() / var) if var > 0 else float("nan"))


def node_nse(pred, true, node_sel, rec_sel):
    """NSE at each node against that node's own mean across snapshots.

    The reference forecast is "this node is always at its average level", so the
    metric asks how much of the between-snapshot variation at that location the
    model explains. Snapshots are treated as independent samples, matching how
    the model is applied; no temporal structure is assumed or used.

    Nodes whose target is essentially constant across the window carry no
    variance to explain; NSE is undefined for them and they are counted
    separately rather than scored as failures.
    """
    p = pred[np.ix_(rec_sel, node_sel)] * 100.0
    t = true[np.ix_(rec_sel, node_sel)] * 100.0
    denom = ((t - t.mean(0, keepdims=True)) ** 2).sum(0)
    sse = ((p - t) ** 2).sum(0)
    live = denom > 1e-6 * max(t.shape[0], 1)
    if live.sum() == 0:
        return dict(n_nodes_scored=0, nse_node_median=float("nan"),
                    nse_node_mean=float("nan"), n_nodes_no_signal=int((~live).sum()))
    nse = 1.0 - sse[live] / denom[live]
    return dict(n_nodes_scored=int(live.sum()),
                n_nodes_no_signal=int((~live).sum()),
                nse_node_median=float(np.median(nse)),
                nse_node_mean=float(nse.mean()),
                nse_node_p10=float(np.percentile(nse, 10)),
                frac_nodes_nse_above_0p5=float((nse > 0.5).mean()))


def bias_noise_split(pred, true, node_sel):
    """Split MAE into a static per-node offset and the part that varies by snapshot."""
    e = (pred[:, node_sel] - true[:, node_sel]) * 100.0
    b = e.mean(0)
    return dict(abs_bias_cm=float(np.abs(b).mean()),
                varying_err_cm=float(e.std(0).mean()),
                bias_share=float(np.abs(b).mean() /
                                 max(np.abs(e).mean(), 1e-9)))


def detection_scores(pred_lvl, true_lvl, thr, sel):
    """POD / FAR / CSI for exceeding a per-node threshold (flooding)."""
    pf = (pred_lvl >= thr[None, :]) & sel
    tf = (true_lvl >= thr[None, :]) & sel
    tp = int((pf & tf).sum()); fp = int((pf & ~tf & sel).sum()); fn = int((~pf & tf).sum())
    return dict(tp=tp, fp=fp, fn=fn,
                pod=float(tp / (tp + fn)) if tp + fn else float("nan"),
                far=float(fp / (tp + fp)) if tp + fp else float("nan"),
                csi=float(tp / (tp + fp + fn)) if tp + fp + fn else float("nan"))


# ── flood geometry ───────────────────────────────────────────────────────────

# Manhole geometry per network. Row order in these CSVs matches manhole node
# order in the cached graphs exactly -- verified by comparing the graph's own
# invert_level_m against the CSV invert column, which agrees to ~1e-6 m on all
# three networks. That check is repeated at runtime and refuses to produce flood
# metrics if it fails, because a silent row-order mismatch would yield plausible
# looking but meaningless numbers.
#
#   icm  (InfoWorks export): invert = chamber_floor, ground = ground_level
#   swmm (SWMM junctions):   invert = elevation,     ground = elevation + max_depth
TOPOLOGY = {
    "heusden": ("data/raw/heusden/Detailed_heusden_topology/"
                "1D rioleringsmodel Heusden!_node.csv", "icm"),
    "bellinge": ("data/interim/bellinge/topology/BellingeSWMM_node.csv", "icm"),
    "tuindorp": ("data/raw/tuindorp/Tuindorp development - "
                 "1 min resolution/networks/Tuindorp development - "
                 "1 min resolution_node.csv", "swmm"),
}
SEARCH_ROOTS = [REPO]


def resolve_topology(data_file: str, override: str | None):
    """Locate the node CSV for a dataset. Returns (path, kind) or None."""
    if override:
        pp = Path(override)
        kind = "swmm" if "swmm" in pp.name.lower() or "tuindorp" in pp.name.lower() \
            else "icm"
        return (pp, kind) if pp.exists() else None
    key = next((k for k in TOPOLOGY if k in str(data_file).lower()), None)
    if key is None:
        return None
    rel, kind = TOPOLOGY[key]
    for root in SEARCH_ROOTS:
        if (root / rel).exists():
            return root / rel, kind
    return None


def flood_geometry(csv_path: Path, kind: str, n: int,
                   invert_raw: np.ndarray, targets: np.ndarray):
    """Per-manhole flood threshold, with unit scale and target frame inferred.

    Returns dict(invert_m, capacity_m, frame, scale, note) or None.

    Two things vary across the three networks and neither is safe to assume:

      scale  The Heusden ICM export writes elevations in feet while its targets
             are in metres; Bellinge's ICM export is already in metres. Both
             candidates are tested rather than hard-coded per network.
      frame  Heusden's target is water level above datum (ICM depnod); the SWMM
             networks' target is depth above invert. Getting this backwards puts
             a whole invert elevation into the error.

    Both are settled by the same physical test: water depth cannot be negative.
    The (scale, frame) pair that yields the fewest negative implied depths wins,
    with median depth below median manhole capacity as the tie-break. This is
    decisive in practice -- the wrong frame puts tens of metres of invert
    elevation into the depth and produces negatives almost everywhere.
    """
    import pandas as pd
    df = pd.read_csv(csv_path, encoding="latin-1")
    tcol = next((c for c in ("node_type", "type") if c in df.columns), None)
    if tcol is None:
        return None
    want = ("manhole",) if kind == "icm" else ("junction", "manhole")
    rows = df[df[tcol].astype(str).str.lower().isin(want)].reset_index(drop=True)
    if len(rows) != n:
        print(f"    [flood] {csv_path.name}: {len(rows)} topology rows vs {n} graph "
              f"nodes - refusing to guess an alignment")
        return None

    if kind == "icm":
        need = ("chamber_floor", "ground_level")
    else:
        need = ("elevation", "max_depth")
    if any(c not in rows.columns for c in need):
        print(f"    [flood] {csv_path.name}: missing {need}")
        return None
    inv_csv = rows[need[0]].astype(float).values
    ground_csv = (rows["ground_level"].astype(float).values if kind == "icm"
                  else inv_csv + rows["max_depth"].astype(float).values)

    # Row order must match graph node order. With --datum_patch_units the graph
    # invert is in metres and the topology column in feet, so resolve the scale first.
    inv_scale, dev = 1.0, float(np.abs(invert_raw - inv_csv).max())
    alt = float(np.abs(invert_raw - inv_csv / FEET_PER_METRE).max())
    if alt < dev:
        inv_scale, dev = FEET_PER_METRE, alt
    if dev > 1e-3:
        print(f"    [flood] {csv_path.name}: invert mismatch {dev:.4g} m at both "
              f"candidate scales - row order does not match graph node order, "
              f"skipping flood metrics")
        return None
    # put the geometry on whatever scale the graph invert is already using
    inv_csv = inv_csv / inv_scale
    ground_csv = ground_csv / inv_scale

    cap_raw = ground_csv - inv_csv
    if not (cap_raw > 0).all():
        print(f"    [flood] {csv_path.name}: {(cap_raw <= 0).sum()} nodes with "
              f"ground level at or below invert")

    sample = targets if targets.shape[0] <= 400 else targets[::max(targets.shape[0]//400, 1)]
    best = None
    # geometry is already on the graph's scale; only the target frame is unknown
    for scale, sname in ((1.0, "graph scale"),):
        cap = cap_raw / scale
        inv_m = inv_csv / scale
        for frame in ("depth", "level"):
            d = sample if frame == "depth" else sample - inv_m[None, :]
            neg = float((d < -0.01).mean())
            med = float(np.median(d))
            if not (0.3 <= float(np.median(cap)) <= 20.0):
                continue
            score = (neg, abs(med) if frame == "depth" else med)
            cand = dict(scale=scale, scale_name=sname, frame=frame,
                        neg_frac=neg, med_depth=med, cap=cap, inv_m=inv_m,
                        med_cap=float(np.median(cap)))
            if best is None or score < (best["neg_frac"], best["_tb"]):
                cand["_tb"] = score[1]
                best = cand
    if best is None:
        return None
    return dict(invert_m=best["inv_m"], capacity_m=best["cap"],
                frame=best["frame"], scale=best["scale"],
                note=f"{best['scale_name']}, target frame = {best['frame']} "
                     f"(negative-depth fraction {best['neg_frac']:.4f}, "
                     f"median manhole capacity {best['med_cap']:.2f} m)",
                med_cap=best["med_cap"], neg_frac=best["neg_frac"])


# ── per-fold analysis ────────────────────────────────────────────────────────

def analyse_fold(k, run_dir, device, flood_quantile, topo_override):
    pred, true, nonsensor, meta = predict_test(run_dir, device)
    n = meta["n_nodes"]
    a = meta["args"]
    scored = np.zeros_like(true, dtype=bool)
    scored[:, nonsensor] = True

    res = {"fold": k, "run_dir": str(run_dir), "ckpt_epoch": meta["ckpt_epoch"],
           "val_mae_cm": meta["val_mae_cm"], "n_test_records": int(true.shape[0]),
           "n_scored_nodes": int(nonsensor.sum()),
           "test_events": sorted(set(meta["test_events"])) if meta["test_events"] else None,
           "datum_mode": a.get("datum_mode", "off"),
           "datum_patch_units": bool(a.get("datum_patch_units", False)),
           "num_blocks": a.get("num_blocks"),
           "learnable_loss_weights": bool(a.get("learnable_loss_weights", False))}

    topo = resolve_topology(a["data_file"], topo_override)
    res["overall"] = elem_metrics(pred, true, scored)
    res["overall"].update(node_nse(pred, true, nonsensor,
                                       np.ones(true.shape[0], dtype=bool)))
    res["overall"].update(bias_noise_split(pred, true, nonsensor))

    # -- flood masks, built in the depth frame
    geo = flood_geometry(*topo, n, meta["invert_ft"], true) if topo else None
    if geo is not None:
        inv_m, cap = geo["invert_m"], geo["capacity_m"]
        # model output -> depth above invert, whichever frame it was trained in
        if a.get("datum_mode", "off") == "depth" or geo["frame"] == "depth":
            depth_pred, depth_true = pred, true
        else:
            depth_pred, depth_true = pred - inv_m[None, :], true - inv_m[None, :]
        thr_depth = cap
        res["flood_threshold"] = {
            "definition": "depth above invert >= manhole capacity (ground - invert)",
            "physical": True, "unit_scale": geo["note"],
            "median_manhole_depth_m": geo["med_cap"],
            "target_frame": geo["frame"]}
    else:
        # No usable topology: fall back to a per-node upper quantile of the target.
        # This marks statistical extremes, not floods, and is labelled as such.
        depth_pred, depth_true = pred, true
        thr_depth = np.quantile(true, flood_quantile, axis=0)
        res["flood_threshold"] = {
            "definition": f"per-node q{flood_quantile:.2f} of target",
            "physical": False, "unit_scale": "n/a",
            "median_manhole_depth_m": None, "target_frame": "unknown"}

    flooded = depth_true >= thr_depth[None, :]
    flood_nodes = flooded.any(0) & nonsensor          # locations that ever flood
    flood_recs = (flooded & scored).any(1)            # scenarios in which any node floods
    res["flood_prevalence"] = {
        "frac_scored_elements_flooded": float((flooded & scored).sum() / max(scored.sum(), 1)),
        "n_flood_nodes": int(flood_nodes.sum()),
        "frac_scored_nodes_ever_flooded": float(flood_nodes.sum() / max(nonsensor.sum(), 1)),
        "n_flood_records": int(flood_recs.sum()),
        "frac_records_with_any_flood": float(flood_recs.mean())}

    # A handful of flooded elements cannot support an NSE; say so rather than
    # printing a number that reads as a measurement.
    MIN_FLOOD = 100
    res["flood"] = {"sufficient_sample": bool((flooded & scored).sum() >= MIN_FLOOD)}
    if flood_nodes.sum():
        res["flood"]["locations"] = elem_metrics(pred, true, scored & flood_nodes[None, :])
        res["flood"]["locations"].update(
            node_nse(pred, true, flood_nodes, np.ones(true.shape[0], dtype=bool)))
    if flood_recs.sum():
        res["flood"]["scenarios"] = elem_metrics(pred, true, scored & flood_recs[:, None])
        res["flood"]["scenarios"].update(node_nse(pred, true, nonsensor, flood_recs))
    if (flooded & scored).sum():
        res["flood"]["elements"] = elem_metrics(pred, true, scored & flooded)
    res["flood"]["detection"] = detection_scores(depth_pred, depth_true, thr_depth, scored)

    # -- control structures
    res["control"] = {}
    grp_nodes = {}
    for grp in ("weir", "pump"):
        if grp not in meta["endpoints"]:
            continue
        us, ds = meta["endpoints"][grp]
        sel = np.zeros(n, dtype=bool)
        sel[us.astype(int)] = True
        sel[ds.astype(int)] = True
        grp_nodes[grp] = sel
    other = nonsensor.copy()
    for sel in grp_nodes.values():
        other &= ~sel
    grp_nodes["neither"] = other

    for grp, sel in grp_nodes.items():
        s = sel & nonsensor
        if s.sum() == 0:
            continue
        m = elem_metrics(pred, true, scored & s[None, :])
        m.update(node_nse(pred, true, s, np.ones(true.shape[0], dtype=bool)))
        m["n_nodes"] = int(s.sum())
        m["mae_over_sigma"] = float(m["mae_cm"] / max(m["sigma_cm"], 1e-9))
        m.update(bias_noise_split(pred, true, s))
        # flood-conditional at control structures: the case the paper cares about
        fs = scored & s[None, :] & flooded
        if fs.sum() >= 50:
            m["flood_elements"] = elem_metrics(pred, true, fs)
        res["control"][grp] = m

    # -- record-level series, for the fold-difference analysis
    err_rec = np.abs((pred - true) * 100.0)
    err_rec = np.where(scored, err_rec, np.nan)
    rec = dict(
        mae_cm=np.nanmean(err_rec, axis=1),
        mean_depth_m=np.nanmean(np.where(scored, depth_true, np.nan), axis=1),
        max_depth_m=np.nanmax(np.where(scored, depth_true, -np.inf), axis=1),
        n_flooded=(flooded & scored).sum(1).astype(float),
        event=np.array(meta["test_events"]) if meta["test_events"]
        else np.array(["?"] * true.shape[0]))

    res["fold_stats"] = {
        "mean_depth_m": float(np.nanmean(rec["mean_depth_m"])),
        "p95_record_mean_depth_m": float(np.nanpercentile(rec["mean_depth_m"], 95)),
        "peak_depth_m": float(np.nanmax(rec["max_depth_m"])),
        "target_snapshot_sigma_cm": float(
            np.nanmean(true[:, nonsensor].std(0)) * 100.0),
        "target_spatial_sigma_cm": float(true[:, nonsensor].mean(0).std() * 100.0),
        "wet_frac": float((rec["mean_depth_m"] >=
                           np.nanquantile(rec["mean_depth_m"], 2.0 / 3.0)).mean()),
        "record_mae_p90_cm": float(np.nanpercentile(rec["mae_cm"], 90))}

    # per-event MAE inside the fold: isolates whether one storm drives the fold
    if meta["test_events"]:
        per_ev = {}
        for e in sorted(set(meta["test_events"])):
            m = rec["event"] == e
            per_ev[e] = dict(n_records=int(m.sum()),
                             mae_cm=float(np.nanmean(rec["mae_cm"][m])),
                             mean_depth_m=float(np.nanmean(rec["mean_depth_m"][m])),
                             peak_depth_m=float(np.nanmax(rec["max_depth_m"][m])))
        res["per_event"] = per_ev

    del pred, true, depth_pred, depth_true, flooded, err_rec
    gc.collect()
    return res, rec


# ── cross-fold synthesis ─────────────────────────────────────────────────────

def spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def explain_folds(folds, recs):
    """Test whether fold MAE spread is explained by storm severity."""
    maes = [f["overall"]["mae_cm"] for f in folds]
    out = {"fold_mae_cm": maes,
           "mean_cm": float(np.mean(maes)), "std_cm": float(np.std(maes, ddof=1)),
           "worst_fold": int(np.argmax(maes)), "best_fold": int(np.argmin(maes))}

    drivers = ["mean_depth_m", "p95_record_mean_depth_m", "peak_depth_m",
               "target_snapshot_sigma_cm", "target_spatial_sigma_cm", "wet_frac"]
    out["rank_corr_with_fold_mae"] = {
        d: spearman([f["fold_stats"][d] for f in folds], maes) for d in drivers}
    out["rank_corr_note"] = ("n=5 folds: these are descriptive only. The pooled "
                             "record-level relation below carries the evidence.")

    # Pooled over every test record in every fold: does |error| rise with depth?
    md = np.concatenate([r["mean_depth_m"] for r in recs])
    ma = np.concatenate([r["mae_cm"] for r in recs])
    ok = np.isfinite(md) & np.isfinite(ma)
    md, ma = md[ok], ma[ok]
    out["pooled_records"] = int(ok.sum())
    out["pooled_corr_depth_vs_record_mae"] = spearman(md, ma)
    qs = np.quantile(md, np.linspace(0, 1, 11))
    curve = []
    for i in range(10):
        m = (md >= qs[i]) & ((md <= qs[i + 1]) if i == 9 else (md < qs[i + 1]))
        if m.sum():
            curve.append(dict(decile=i + 1, mean_depth_m=float(md[m].mean()),
                              mae_cm=float(ma[m].mean()), n=int(m.sum())))
    out["depth_decile_curve"] = curve

    # Predict each fold's MAE from the pooled curve alone: if the residual is
    # small, the fold is unremarkable given the storms it happened to hold out.
    cd = np.array([c["mean_depth_m"] for c in curve])
    cm = np.array([c["mae_cm"] for c in curve])
    pred_fold, resid = [], []
    for f, r in zip(folds, recs):
        d = r["mean_depth_m"][np.isfinite(r["mean_depth_m"])]
        exp = float(np.interp(d, cd, cm).mean())
        pred_fold.append(exp)
        resid.append(f["overall"]["mae_cm"] - exp)
    out["expected_mae_from_depth_curve_cm"] = pred_fold
    out["residual_after_depth_cm"] = resid
    out["frac_spread_explained_by_depth"] = float(
        1.0 - np.var(resid, ddof=1) / max(np.var(maes, ddof=1), 1e-12))
    return out


def pooled_block(folds, key, sub=None):
    vals = []
    for f in folds:
        d = f.get(key) if sub is None else f.get(key, {}).get(sub)
        if d:
            vals.append(d)
    if not vals:
        return None
    out = {}
    for m in ("mae_cm", "rmse_cm", "nse_pooled", "nse_node_median",
              "mae_over_sigma", "sigma_cm", "csi", "pod", "far"):
        xs = [v[m] for v in vals if m in v and np.isfinite(v[m])]
        if xs:
            out[m] = float(np.mean(xs))
            out[m + "_std"] = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    out["n_folds"] = len(vals)
    return out


# ── reporting ────────────────────────────────────────────────────────────────

def report(folds, cross, tag):
    p = print
    p(f"\n{'='*78}\n TEST-SPLIT BREAKDOWN — {tag}\n{'='*78}")

    p("\n1. OVERALL (held-out test split, non-sensor manholes only)")
    p(f"{'fold':>4} {'events':>9} {'ep':>4} {'MAE':>7} {'RMSE':>7} "
      f"{'NSEpool':>8} {'NSEnode':>8} {'bias':>7} {'noise':>7} {'bias%':>6}")
    for f in folds:
        o = f["overall"]
        ev = ",".join(f["test_events"]) if f["test_events"] else "?"
        p(f"{f['fold']:>4} {ev:>9} {f['ckpt_epoch']:>4} {o['mae_cm']:7.3f} "
          f"{o['rmse_cm']:7.3f} {o['nse_pooled']:8.4f} {o['nse_node_median']:8.4f} "
          f"{o['abs_bias_cm']:7.3f} {o['varying_err_cm']:7.3f} "
          f"{100*o['bias_share']:5.0f}%")
    m = [f["overall"]["mae_cm"] for f in folds]
    np_ = [f["overall"]["nse_pooled"] for f in folds]
    nt = [f["overall"]["nse_node_median"] for f in folds]
    p(f"{'mean':>4} {'':>9} {'':>4} {np.mean(m):7.3f} "
      f"{np.mean([f['overall']['rmse_cm'] for f in folds]):7.3f} "
      f"{np.mean(np_):8.4f} {np.mean(nt):8.4f}")
    p(f"{'sd':>4} {'':>9} {'':>4} {np.std(m, ddof=1):7.3f}")
    p("\n  NSEpool pools one variance over all (snapshot, node) elements. On a level "
      "target\n  that denominator is mostly static elevation spread between nodes, "
      "which the\n  model receives as an input feature. NSEnode scores each node "
      "against its own\n  mean across snapshots (median over nodes) and is the "
      "number to quote for\n  predicting hydraulic state.")

    p("\n2. WHY THE FOLDS DIFFER")
    p(f"{'fold':>4} {'MAE':>7} {'meanD':>7} {'peakD':>7} {'sigT':>7} {'sigS':>8} "
      f"{'fld%':>6} {'expMAE':>7} {'resid':>7}")
    for i, f in enumerate(folds):
        s = f["fold_stats"]
        p(f"{f['fold']:>4} {f['overall']['mae_cm']:7.3f} {s['mean_depth_m']:7.3f} "
          f"{s['peak_depth_m']:7.2f} {s['target_snapshot_sigma_cm']:7.2f} "
          f"{s['target_spatial_sigma_cm']:8.1f} "
          f"{100*f['flood_prevalence']['frac_scored_elements_flooded']:5.2f}% "
          f"{cross['expected_mae_from_depth_curve_cm'][i]:7.3f} "
          f"{cross['residual_after_depth_cm'][i]:+7.3f}")
    p(f"\n  pooled record-level rank corr(mean depth, record MAE) = "
      f"{cross['pooled_corr_depth_vs_record_mae']:+.3f} over "
      f"{cross['pooled_records']} records")
    p(f"  fraction of between-fold MAE variance explained by storm depth alone = "
      f"{100*cross['frac_spread_explained_by_depth']:.0f}%")
    p("  per-fold rank correlations (n=5, descriptive only):")
    for k, v in cross["rank_corr_with_fold_mae"].items():
        p(f"    {k:<28s} {v:+.2f}")
    for f in folds:
        if "per_event" in f:
            ev = "  ".join(f"{e}:{d['mae_cm']:.2f}cm(D={d['mean_depth_m']:.2f}m)"
                           for e, d in f["per_event"].items())
            p(f"    fold {f['fold']} by storm: {ev}")

    fl = folds[0]["flood_threshold"]
    if fl.get("physical"):
        p("\n3. FLOODING")
    else:
        p("\n3. EXTREMES (no ground level in this dataset — NOT flooding)")
        p("   The topology carries no ground/rim level, so surcharge to the surface")
        p("   cannot be identified. The threshold below is a statistical upper")
        p("   quantile of each node's own target: it marks extreme levels, not")
        p("   floods, and must not be reported as flood skill.")
    p(f"  definition: {fl['definition']}  [{fl['unit_scale']}]")
    if fl["median_manhole_depth_m"]:
        p(f"  median manhole depth (ground - invert) = "
          f"{fl['median_manhole_depth_m']:.2f} m")
    p(f"\n{'fold':>4} {'fldElem%':>9} {'fldNodes':>9} {'fldRecs':>8} | "
      f"{'MAE_loc':>8} {'NSE_loc':>8} | {'MAE_scen':>9} | {'MAE_elem':>9} | "
      f"{'POD':>5} {'FAR':>5} {'CSI':>5}")
    for f in folds:
        pv, fd = f["flood_prevalence"], f["flood"]
        loc, sc, el, dt = (fd.get("locations"), fd.get("scenarios"),
                           fd.get("elements"), fd["detection"])
        p(f"{f['fold']:>4} {100*pv['frac_scored_elements_flooded']:8.3f}% "
          f"{pv['n_flood_nodes']:9d} {pv['n_flood_records']:8d} | "
          f"{loc['mae_cm'] if loc else float('nan'):8.3f} "
          f"{loc['nse_node_median'] if loc else float('nan'):8.4f} | "
          f"{sc['mae_cm'] if sc else float('nan'):9.3f} | "
          f"{el['mae_cm'] if el else float('nan'):9.3f} | "
          f"{dt['pod']:5.2f} {dt['far']:5.2f} {dt['csi']:5.2f}")
    w = "flood" if fl.get("physical") else "exceedance"
    p(f"  loc = nodes with any {w}; scen = records with any {w};")
    p(f"  elem = the individual {w} (time, node) entries.")

    p("\n4. CONTROL STRUCTURES (weir / pump adjacency)")
    p(f"{'fold':>4} " + " ".join(f"{g+'_mae':>10} {g+'_nse':>9} {g+'_m/s':>7}"
                                 for g in ("weir", "pump", "neither")))
    for f in folds:
        row = f"{f['fold']:>4} "
        for g in ("weir", "pump", "neither"):
            c = f["control"].get(g)
            row += (f"{c['mae_cm']:10.3f} {c['nse_node_median']:9.4f} "
                    f"{c['mae_over_sigma']:7.3f} " if c else f"{'-':>28} ")
        p(row)
    c0 = folds[0]["control"]
    p("  n nodes: " + ", ".join(f"{g}={c0[g]['n_nodes']}" for g in c0))
    p("  m/s = MAE / target sigma in the same group. Weir-adjacent nodes are more")
    p("  variable by construction, so compare m/s, not MAE, across groups.")
    for g in ("weir", "pump"):
        fe = [f["control"][g]["flood_elements"]["mae_cm"]
              for f in folds if f["control"].get(g, {}).get("flood_elements")]
        if fe:
            p(f"  {g}-adjacent MAE on flooded elements: {np.mean(fe):.2f} cm "
              f"(mean of {len(fe)} folds)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help="run dir template containing {k}, e.g. outputs/x_fold{k}")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--flood_quantile", type=float, default=0.99,
                    help="fallback per-node threshold when no ground level exists")
    ap.add_argument("--topology", default=None,
                    help="node CSV override; otherwise resolved from the dataset name")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tag = args.tag or Path(args.runs.replace("{k}", "")).name.rstrip("_")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {dev}")

    folds, recs = [], []
    for k in args.folds:
        rd = REPO / args.runs.format(k=k)
        if not (rd / "best_model.pth").exists():
            print(f"[skip] fold {k}: no checkpoint at {rd}")
            continue
        print(f"[fold {k}] {rd}")
        r, rec = analyse_fold(k, rd, dev, args.flood_quantile, args.topology)
        print(f"    MAE {r['overall']['mae_cm']:.3f} cm  "
              f"NSEpool {r['overall']['nse_pooled']:.4f}  "
              f"NSEnode {r['overall']['nse_node_median']:.4f}")
        folds.append(r); recs.append(rec)
        gc.collect()

    if not folds:
        raise SystemExit("no folds analysed")

    cross = explain_folds(folds, recs)
    report(folds, cross, tag)

    out = {"tag": tag, "folds": folds, "fold_comparison": cross,
           "pooled": {"overall": pooled_block(folds, "overall"),
                      "flood_locations": pooled_block(folds, "flood", "locations"),
                      "flood_scenarios": pooled_block(folds, "flood", "scenarios"),
                      "flood_elements": pooled_block(folds, "flood", "elements"),
                      "flood_detection": pooled_block(folds, "flood", "detection"),
                      "weir": pooled_block(folds, "control", "weir"),
                      "pump": pooled_block(folds, "control", "pump"),
                      "neither": pooled_block(folds, "control", "neither")}}
    dest = Path(args.out) if args.out else REPO / f"results/fold_breakdown_{tag}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, default=float))
    print(f"\nWritten {dest}")


if __name__ == "__main__":
    main()
