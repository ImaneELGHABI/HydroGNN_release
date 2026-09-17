"""Table 2, HydroGNN rows, and the NSE values quoted in Section 5.

Heusden values are means of the per-fold statistics.

    python -m analysis.table2_hydrognn
"""
import json

import numpy as np

from analysis.common import RESULTS, diagnostics, fold_lengths, load_predictions

rows = {}
for net in ["bellinge", "tuindorp", "heusden"]:
    pred, true, ns = load_predictions(net)
    if net == "heusden":
        b = np.cumsum([0] + fold_lengths())
        per_fold = [diagnostics(pred[b[i]:b[i + 1]], true[b[i]:b[i + 1]], ns) for i in range(5)]
        rows[net] = {m: float(np.mean([f[m] for f in per_fold])) for m in per_fold[0]}
        rows[net]["mae_sd"] = float(np.std([f["mae"] for f in per_fold], ddof=1))
        rows[net]["mae_folds"] = [f["mae"] for f in per_fold]
    else:
        rows[net] = diagnostics(pred, true, ns)

print(f"{'':9s} {'MAE':>6s} {'Wet':>6s} {'RMSE':>6s} {'DynR':>5s} {'rho':>5s} {'NSEpool':>7s} {'NSEnode':>7s}")
for net, r in rows.items():
    print(f"{net:9s} {r['mae']:6.2f} {r['wet_mae']:6.2f} {r['rmse']:6.2f} {r['dyn_ratio']:5.2f} "
          f"{r['rho']:5.2f} {r['nse_pooled']:7.3f} {r['nse_node']:7.3f}")
RESULTS.mkdir(exist_ok=True)
(RESULTS / "table2_hydrognn.json").write_text(json.dumps(rows, indent=1))
