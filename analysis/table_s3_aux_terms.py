"""Table S3: auxiliary-term variants on Heusden at 1% (validation MAE, cm),
and the explicit weir/pump terms of Section S8.2 on folds 0 and 4.

    python -m analysis.table_s3_aux_terms
"""
import json

import numpy as np

from analysis.common import OUT, RESULTS


def val(run):
    return json.loads((run / "test_metrics.json").read_text())["best_val_mae_cm"]


rows = {
    "no auxiliary terms": [OUT / f"ablations/no_aux_fold{k}" for k in range(5)],
    "reported configuration": [OUT / f"hydrognn/heusden_1pct_fold{k}" for k in range(5)],
    "no warm-up ramp": [OUT / f"ablations/no_warmup_fold{k}" for k in range(5)],
    "learned term weights": [OUT / f"ablations/learned_fold{k}" for k in range(5)],
}
out = {}
for name, runs in rows.items():
    v = [val(r) for r in runs]
    out[name] = v
    print(f"{name:24s} " + " ".join(f"{x:6.2f}" for x in v) + f"   mean {np.mean(v):.2f}")

print("\nweir/pump terms (fold 0, fold 4); control = no auxiliary terms")
for tag in ["full", "unsup", "strong"]:
    v = [val(OUT / f"ablations/hydraulic_{tag}_fold{k}") for k in (0, 4)]
    out[f"hydraulic_{tag}"] = v
    print(f"  {tag:7s} {v[0]:6.2f} {v[1]:6.2f}")
print(f"  control {out['no auxiliary terms'][0]:6.2f} {out['no auxiliary terms'][4]:6.2f}")

RESULTS.mkdir(exist_ok=True)
(RESULTS / "table_s3_aux_terms.json").write_text(json.dumps(out, indent=1))
