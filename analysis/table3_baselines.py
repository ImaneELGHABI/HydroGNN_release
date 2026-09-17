"""Table 3 and the margins in Sections 4.2 and 4.4.

Heusden: mean over folds 0-4 with the sample standard deviation.

    python -m analysis.table3_baselines
"""
import json

import numpy as np

from analysis.common import OUT, RESULTS, load_predictions

ARCHS = ["gin", "sage", "gcn", "gat", "gatres"]


def base(path):
    j = json.loads((path / "test_metrics.json").read_text())
    return j["mae"], j["rmse"], j["pearson_r"]


def hydro(run):
    o = json.loads((OUT / "hydrognn" / run / "test_metrics.json").read_text())["overall"]
    return o["mae_cm"], o["rmse_cm"]


def pooled_r(net):
    p, t, ns = load_predictions(net)
    p, t = p[:, ns].ravel(), t[:, ns].ravel()
    return float(np.corrcoef(p, t)[0, 1])


table = {}
h = [hydro(f"heusden_1pct_fold{k}") for k in range(5)]
table["heusden"] = {"hydrognn": dict(mae=np.mean([x[0] for x in h]), sd=np.std([x[0] for x in h], ddof=1),
                                     rmse=np.mean([x[1] for x in h]), folds=[x[0] for x in h])}
for a in ARCHS:
    r = [base(OUT / f"baselines/heusden_fold{k}/{a}") for k in range(5)]
    table["heusden"][a] = dict(mae=np.mean([x[0] for x in r]), sd=np.std([x[0] for x in r], ddof=1),
                               rmse=np.mean([x[1] for x in r]), folds=[x[0] for x in r])
for net in ["bellinge", "tuindorp"]:
    m, r = hydro(f"{net}_1pct")
    table[net] = {"hydrognn": dict(mae=m, rmse=r, pooled_r=pooled_r(net))}
    for a in ARCHS:
        mae, rmse, pr = base(OUT / f"baselines/{net}/{a}")
        table[net][a] = dict(mae=mae, rmse=rmse, pooled_r=pr)

for net, d in table.items():
    print(net)
    for k, v in d.items():
        sd = f" +/- {v['sd']:.2f}" if "sd" in v else ""
        r = f"  r {v['pooled_r']:.2f}" if "pooled_r" in v else ""
        print(f"  {k:9s} MAE {v['mae']:6.2f}{sd:10s} RMSE {v['rmse']:6.2f}{r}")
    bl = {k: v for k, v in d.items() if k != "hydrognn"}
    hm = d["hydrognn"]["mae"]
    best, worst = min(bl, key=lambda k: bl[k]["mae"]), max(bl, key=lambda k: bl[k]["mae"])
    print(f"  vs strongest ({best}): {bl[best]['mae'] - hm:.2f} cm, {100 * (1 - hm / bl[best]['mae']):.1f}%")
    print(f"  vs weakest ({worst}): {100 * (1 - hm / bl[worst]['mae']):.1f}%")
    if net == "heusden":
        wins = all(d["hydrognn"]["folds"][k] < v["folds"][k] for v in bl.values() for k in range(5))
        print(f"  lower MAE than every baseline on every fold: {wins}")

RESULTS.mkdir(exist_ok=True)
(RESULTS / "table3_baselines.json").write_text(json.dumps(table, indent=1, default=float))
