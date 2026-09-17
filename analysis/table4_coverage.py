"""Table 4 (coverage sweep) and Table S2 (per-fold values).

    python -m analysis.table4_coverage
"""
import json

import numpy as np

from analysis.common import OUT, RESULTS

COVS = [1, 5, 10, 20]


def test_mae(run):
    return json.loads((OUT / "hydrognn" / run / "test_metrics.json").read_text())["overall"]["mae_cm"]


def val_mae(run):
    return json.loads((OUT / "hydrognn" / run / "test_metrics.json").read_text())["best_val_mae_cm"]


out = {"heusden": {}, "bellinge": {}, "tuindorp": {}}
print(f"{'cov':>4s} {'Heusden mean':>16s} {'fold 4':>7s} {'Bellinge':>9s} {'Tuindorp':>9s}")
for c in COVS:
    f = [test_mae(f"heusden_{c}pct_fold{k}") for k in range(5)]
    out["heusden"][c] = dict(mean=np.mean(f), sd=np.std(f, ddof=1), folds=f,
                             val_folds=[val_mae(f"heusden_{c}pct_fold{k}") for k in range(5)])
    out["bellinge"][c] = test_mae(f"bellinge_{c}pct")
    out["tuindorp"][c] = test_mae(f"tuindorp_{c}pct")
    print(f"{c:3d}% {out['heusden'][c]['mean']:9.2f} +/- {out['heusden'][c]['sd']:4.2f} "
          f"{f[4]:7.2f} {out['bellinge'][c]:9.2f} {out['tuindorp'][c]:9.2f}")

h = out["heusden"]
print(f"\n1% -> 20% change in the Heusden mean: {h[1]['mean'] - h[20]['mean']:.2f} cm")
for k in range(5):
    print(f"  fold {k}: {h[1]['folds'][k]:.2f} -> {h[20]['folds'][k]:.2f}")
t = out["tuindorp"]
print(f"Tuindorp 1% -> 5%: {100 * (1 - t[5] / t[1]):.1f}% lower MAE")

print("\nTable S2 (validation MAE, cm)")
for c in COVS:
    print(f"  Heusden {c:2d}%  " + "  ".join(f"{v:5.2f}" for v in h[c]["val_folds"])
          + f"   mean {np.mean(h[c]['val_folds']):.2f}")
for net in ["bellinge", "tuindorp"]:
    v = [val_mae(f"{net}_{c}pct") for c in COVS]
    out[f"{net}_val"] = dict(zip(COVS, v))
    print(f"  {net:9s}" + "  ".join(f"{c}%: {x:.2f}" for c, x in zip(COVS, v)))

RESULTS.mkdir(exist_ok=True)
(RESULTS / "table4_coverage.json").write_text(json.dumps(out, indent=1, default=float))
