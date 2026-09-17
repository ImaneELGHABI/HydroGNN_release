"""Section 4.5: variability-normalised error and sensor reachability.

Normalised error = per-node MAE / per-node temporal std of the target over the
test split. Group statistics are medians over non-sensor manholes; manholes
whose target is constant over the split are left out.

A manhole is unreachable when its connected component (manholes, storage and
outfalls joined by every link type) contains no sensor.

    python -m analysis.spatial_error
"""
import glob
import json
import os
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.common import PRED, RESULTS
from figures.common import COND_DIR, MH_TYPE, TOPO

SIGMA_FLOOR = 1.0   # cm; the normalised error is not used below this std


def node_graph(net):
    """Full node graph (manholes + storage + outfalls as routers)."""
    meta = pd.read_csv(TOPO[net], encoding="latin-1")
    tc = next(c for c in meta.columns if c.lower() in ("node_type", "type"))
    ids = [str(v) for v in meta["node_id"].values]
    pos = {k: i for i, k in enumerate(ids)}
    is_mh = np.array([t == MH_TYPE[net] for t in meta[tc].values])
    mh_rows = np.where(is_mh)[0]

    adj = [[] for _ in ids]
    seg = []
    xy = meta[["x", "y"]].values.astype(float)
    for lf in sorted(glob.glob(os.path.join(COND_DIR[net], "*.csv"))):
        if "_node" in os.path.basename(lf).lower():
            continue
        df = pd.read_csv(lf, encoding="latin-1")
        if "us_node_id" not in df.columns:
            continue
        for u, v in zip(df["us_node_id"].astype(str), df["ds_node_id"].astype(str)):
            iu, iv = pos.get(u, -1), pos.get(v, -1)
            if iu >= 0 and iv >= 0:
                adj[iu].append(iv); adj[iv].append(iu)
                seg.append([tuple(xy[iu]), tuple(xy[iv])])

    comp = np.full(len(ids), -1)
    c = 0
    for s in range(len(ids)):
        if comp[s] >= 0:
            continue
        q = deque([s]); comp[s] = c
        while q:
            u = q.popleft()
            for v in adj[u]:
                if comp[v] < 0:
                    comp[v] = c; q.append(v)
        c += 1
    return meta, mh_rows, comp, c, seg, xy


def load(net):
    meta, mh_rows, comp, ncomp, _, _ = node_graph(net)
    d = np.load(PRED / f"{net}.npz", allow_pickle=True)
    scored = d["non_sensor"]
    mae, sig = d["mae_node_cm"].astype(float), d["true_std_node_cm"].astype(float)
    sensor_comps = set(comp[mh_rows[~scored]])
    covered = np.array([comp[r] in sensor_comps for r in mh_rows])
    return dict(meta=meta, mh_rows=mh_rows, comp=comp, ncomp=ncomp, d=d,
                scored=scored, mae=mae, sig=sig, covered=covered,
                sensor_comps=sensor_comps)


def reachability(net):
    D = load(net)
    s, cov, sig, mae = D["scored"], D["covered"], D["sig"], D["mae"]
    norm = np.where(sig > SIGMA_FLOOR, mae / np.maximum(sig, 1e-9), np.nan)
    med = lambda m: float(np.nanmedian(norm[m & s])) if (m & s).any() else float("nan")
    comp_mh = D["comp"][D["mh_rows"]]
    res = dict(n_components=int(D["ncomp"]), n_components_with_sensor=len(D["sensor_comps"]),
               n_scored=int(s.sum()), n_unreachable=int((s & ~cov).sum()),
               pct_unreachable=float(100 * (s & ~cov).sum() / s.sum()),
               norm_err_reachable=med(cov), norm_err_unreachable=med(~cov),
               sigma_median=float(np.median(sig[s])),
               sigma_p99=float(np.percentile(sig[s], 99)),
               pct_sigma_below_1cm=float(100 * (sig[s] < 1).mean()))

    # where the sensors sit
    sens_comp = comp_mh[~s]
    biggest = max(set(sens_comp), key=lambda c: (comp_mh == c).sum())
    res["largest_sensor_component_manholes"] = int((comp_mh == biggest).sum())
    res["sensors_in_largest_component"] = int((sens_comp == biggest).sum())

    # unreachable manholes, per component
    u = s & ~cov
    sizes = pd.Series(comp_mh[u]).value_counts()
    res["unreachable_components"] = int(len(sizes))
    res["largest_unreachable_component"] = int(sizes.max()) if len(sizes) else 0
    res["unreachable_in_largest_nine"] = int(sizes.iloc[:9].sum()) if len(sizes) else 0

    meta = D["meta"]
    if "system_type" in meta.columns:
        st = meta["system_type"].astype(str).str.lower().values[D["mh_rows"]]
        res["by_system_type"] = {}
        for t in sorted(set(st)):
            m = st == t
            res["by_system_type"][t] = dict(
                n=int((m & s).sum()),
                pct_unreachable=float(100 * (m & u).sum() / max((m & s).sum(), 1)),
                norm_err=med(m),
                norm_err_reachable=med(m & cov),
                norm_err_unreachable=med(m & ~cov))
    return res


def variability(net):
    D = load(net)
    s, sig, mae, d = D["scored"], D["sig"], D["mae"], D["d"]
    ok = s & (sig > 0)
    norm = np.where(ok, mae / np.maximum(sig, 1e-12), np.nan)
    dyn = d["pred_std_node_cm"] / np.maximum(sig, 1e-12)
    nse = d["nse_node"]
    quiet = ok & (sig < np.median(sig[s]))
    active = ok & (sig >= np.percentile(sig[s], 90))
    idx = np.where(ok)[0]
    worst_abs = set(idx[np.argsort(-mae[ok])[:100]])
    worst_norm = set(idx[np.argsort(-norm[ok])[:100]])
    return dict(norm_err_quiet_half=float(np.median(norm[quiet])),
                norm_err_dynamic_decile=float(np.median(norm[active])),
                dyn_ratio_quiet_half=float(np.median(dyn[quiet])),
                dyn_ratio_dynamic_decile=float(np.median(dyn[active])),
                nse_quiet_half=float(np.nanmedian(nse[quiet])),
                nse_dynamic_decile=float(np.nanmedian(nse[active])),
                spearman_abs_vs_norm=float(spearmanr(mae[ok], norm[ok]).correlation),
                worst100_shared=len(worst_abs & worst_norm))


if __name__ == "__main__":
    out = {net: reachability(net) for net in ("bellinge", "heusden")}
    out["bellinge"]["variability"] = variability("bellinge")
    print(json.dumps(out, indent=1, default=float))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "section4_5_spatial.json").write_text(json.dumps(out, indent=1, default=float))
