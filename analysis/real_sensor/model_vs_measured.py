"""Section 5.1 and Table 5: HydroGNN vs the measured level at G72F040.

The graphs carry no timestamp, so each is matched to its (event, timestep) by
comparing its target vector with the simulation CSVs.

The cleaned record ends 2020-03-23, so these events belong to the validation
period (test events are from 2021). One sensor only.

    python -m analysis.real_sensor.model_vs_measured
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from hydrognn.data_loader import lines_as_nodes                        # noqa: E402
from hydrognn.model import HydroGNN, schema_from_graph                 # noqa: E402
from hydrognn.train import (add_sensor_context_features, apply_mask,   # noqa: E402
                            build_fixed_mask, compute_climatology,
                            load_sensor_indices, load_system_type_idx)
from torch_geometric.loader import DataLoader                          # noqa: E402

SENSOR = NODE_ID = "G72F040"
RUN = REPO / "outputs/hydrognn/bellinge_1pct"
DATA = REPO / "data/processed/bellinge_v3_nopumpstatus.pt"
CLEANED = REPO / "data/raw/bellinge/2_cleaned_data/G72F040_Danovap1_proc_v6.csv"
SIMDIR = REPO / "data/interim/bellinge/simulations"
TOPO = REPO / "data/interim/bellinge/topology/BellingeSWMM_node.csv"


def manhole_ids():
    b = pd.read_csv(TOPO, encoding="latin-1")
    mh = b[b.node_type.astype(str).str.lower() == "manhole"].reset_index(drop=True)
    return mh["node_id"].astype(str).tolist()


def fingerprint(Y, invert, ids, tags):
    """graph index -> timestamp, by matching target vectors against simulation rows.

    A brute-force scan would be 31 events x ~4000 rows x 20,535 graphs x 995 dims,
    about 2.5 trillion element operations. Instead each graph is reduced to a short
    signature -- the targets at the few manholes that vary most across the dataset
    -- and a KD-tree over those signatures turns every row into one nearest-neighbour
    query. The candidate is then verified on the full 995-dim vector, so the
    signature only has to narrow the field, not decide the match.
    """
    from scipy.spatial import cKDTree
    k = 16
    var = Y.var(0)
    sig_cols = np.argsort(var)[-k:]                  # most discriminative manholes
    tree = cKDTree(Y[:, sig_cols])
    stamp, checked = {}, 0
    for tag in tags:
        f = glob.glob(str(SIMDIR / f"*{tag}_depnod.csv"))
        if not f:
            continue
        sim = pd.read_csv(f[0], skiprows=[1], low_memory=False)
        cols = [c for c in ids if c in sim.columns]
        if len(cols) < len(ids) * 0.99:
            continue
        order = [ids.index(c) for c in cols]
        S = sim[cols].apply(pd.to_numeric, errors="coerce").values
        D = S - invert[order][None, :]                # level -> depth
        t = pd.to_datetime(sim["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")

        # map signature columns (graph order) into this file's column order
        pos = {n: j for j, n in enumerate(order)}
        if not all(c in pos for c in sig_cols):
            continue
        Dsig = D[:, [pos[c] for c in sig_cols]]
        good = np.isfinite(Dsig).all(1) & np.isfinite(D).all(1) & t.notna().values
        if not good.any():
            continue
        _, cand = tree.query(Dsig[good], k=1, workers=-1)
        rows = np.where(good)[0]
        Yo = Y[:, order]
        for r, i in zip(rows, np.atleast_1d(cand)):
            i = int(i)
            if i in stamp:
                continue
            checked += 1
            if np.abs(Yo[i] - D[r]).mean() < 1e-4:    # verify on the full vector
                stamp[i] = t.iloc[r]
    print(f"  signature dims={k}, candidates verified={checked}, accepted={len(stamp)}")
    return stamp


def M(x, y):
    e = x - y
    var = ((y - y.mean()) ** 2).sum()
    return dict(n=int(len(e)), mae_cm=float(np.abs(e).mean() * 100),
                rmse_cm=float(np.sqrt((e ** 2).mean()) * 100),
                bias_cm=float(e.mean() * 100),
                nse=float(1 - (e ** 2).sum() / var) if var > 0 else float("nan"),
                r=float(np.corrcoef(x, y)[0, 1]))


def main():
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw = torch.load(DATA, map_location="cpu", weights_only=False)
    dmean, dstd = float(raw["depth_mean"]), float(raw["depth_std"])
    ids = manhole_ids(); node = ids.index(NODE_ID)
    ck = torch.load(RUN / "best_model.pth", map_location="cpu", weights_only=False)
    a = ck["args"]
    sens = load_sensor_indices(REPO / "data/sensors" / Path(a["sensor_file"]).name)
    assert node not in sens, "target must be masked, not an input sensor"
    print(f"target {SENSOR} -> manhole {node}; model sees {len(sens)} sensors (target masked)")

    obs = pd.read_csv(CLEANED, usecols=["time", "level"], parse_dates=["time"])
    obs = pd.Series(pd.to_numeric(obs["level"], errors="coerce").values,
                    index=obs["time"]).dropna()
    obs = obs[~obs.index.duplicated()]
    ev = pd.read_csv(REPO / "data/interim/bellinge/event_list.csv",
                     parse_dates=["start", "end"])
    win = ev[(ev.start >= obs.index.min().normalize()) & (ev.end <= obs.index.max())]

    inv = raw["graphs"][0]["manhole"].invert_level_m.view(-1).numpy()
    Y = np.stack([g["manhole"].y.view(-1).numpy() for g in raw["graphs"]]) * dstd + dmean
    stamp = fingerprint(Y, inv, ids, win.tag.tolist())
    print(f"events in window {len(win)}; graphs matched to a timestamp: {len(stamp)}")
    if not stamp:
        raise SystemExit("fingerprinting failed")
    idx = sorted(stamp)

    qmax = a.get("add_qmax_feature", False); sc = a.get("manhole_shortcut", False)
    na = a.get("normalize_assets", False); n_tr = int(raw["train_size"])
    train_g = [lines_as_nodes(g, qmax, sc, na) for g in raw["graphs"][:n_tr]]
    sel = [lines_as_nodes(raw["graphs"][i], qmax, sc, na) for i in idx]
    n = train_g[0]["manhole"].x.size(0); nf = train_g[0]["manhole"].x.size(1)
    sfi, didx = nf - 1, list(range(3, nf - 1))
    mask = build_fixed_mask(n, sens)
    sel = [apply_mask(g, mask, didx, sfi) for g in sel]
    if a.get("wetness_features"):
        cmask = None if a.get("leaky_climatology", True) else ~mask
        cmu, csd = compute_climatology([apply_mask(g, mask, didx, sfi) for g in train_g],
                                       sensor_mask=cmask)
        sel = [add_sensor_context_features(g, cmu, csd, didx[0],
                                           include_global=a.get("global_context", False))
               for g in sel]
    del train_g
    nd, ed, ets = schema_from_graph(sel[0])
    si = (load_system_type_idx(TOPO, n)
          if a.get("use_system_type") else None)
    model = HydroGNN(node_dims=nd, edge_dims=ed, edge_types=ets,
                     hidden_dim=a["hidden_dim"], num_blocks=a["num_blocks"],
                     heads=a["heads"], dropout=a["dropout"],
                     use_softplus_output=a["softplus_output"],
                     use_system_type_conditioning=a.get("use_system_type", False),
                     use_input_skip=a.get("input_skip", False)).to(dev)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    P = []
    with torch.no_grad():
        for b in DataLoader(sel, batch_size=a["batch_size"], shuffle=False):
            b = b.to(dev)
            ng = b["manhole"].x.size(0) // n
            sti = si.to(dev).repeat(ng) if si is not None else None
            ea = {e: (b[e].edge_attr if "edge_attr" in b[e] else None) for e in b.edge_types}
            pr = model(b.x_dict, b.edge_index_dict, ea, system_type_idx=sti)
            P.append((pr * dstd + dmean).view(ng, n)[:, node].cpu().numpy())
    pred_level = np.concatenate(P) + inv[node]          # SWMM target is depth
    sim_level = Y[idx, node] + inv[node]

    ts = pd.to_datetime([stamp[i] for i in idx])
    j = obs.reindex(ts, method="nearest", tolerance=pd.Timedelta("1min"))
    ok = j.notna().values
    o = j.values[ok].astype(float); p = pred_level[ok]; s_ = sim_level[ok]
    print(f"snapshots joined to a measurement: {int(ok.sum())} of {len(idx)}\n")

    mm, ms, msm = M(p, o), M(s_, o), M(p, s_)
    print(f"{'comparison':<34}{'MAE':>8}{'RMSE':>8}{'bias':>9}{'NSE':>8}{'r':>7}")
    for lab, m in (("HydroGNN vs MEASUREMENT", mm),
                   ("simulator vs MEASUREMENT", ms),
                   ("HydroGNN vs simulator", msm)):
        print(f"{lab:<34}{m['mae_cm']:8.2f}{m['rmse_cm']:8.2f}{m['bias_cm']:+9.2f}"
              f"{m['nse']:8.3f}{m['r']:7.3f}")
    rng = 100 * (o.max() - o.min())
    # NSE after removing the mean offset, which the model inherits from the simulator
    q = p - (p - o).mean()
    nse_no_offset = float(1 - ((q - o) ** 2).sum() / ((o - o.mean()) ** 2).sum())
    print(f"\nmeasured range {rng:.1f} cm over {mm['n']} timesteps")
    print(f"HydroGNN vs measurement, mean offset removed: NSE {nse_no_offset:.3f}")
    out = REPO / "results/real_sensor_model_vs_measured.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"sensor": SENSOR, "node": node, "n": mm["n"],
                               "measured_range_cm": float(rng),
                               "model_vs_measured": mm, "sim_vs_measured": ms,
                               "model_vs_sim": msm,
                               "model_vs_measured_nse_offset_removed": nse_no_offset}, indent=2))
    print(f"Written {out}")


if __name__ == "__main__":
    main()
