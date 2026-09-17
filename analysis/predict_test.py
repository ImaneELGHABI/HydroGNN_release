"""Held-out test predictions of the 1% HydroGNN runs.

Writes outputs/predictions/{net}.npz with preds, trues (cm, [records, manholes]),
the non-sensor mask and per-node metrics. Heusden concatenates folds 0-4 in
order, so every design storm appears once.

    python -m analysis.predict_test [heusden bellinge tuindorp]
"""
import gc
import sys

import numpy as np
import torch

from analysis.common import (NETWORKS, PRED, TOPOLOGY, globals_, per_node,
                             predict_test, sensor_file)


def build(net: str) -> None:
    cfg = NETWORKS[net]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    P, Y, mask = [], [], None
    for k in cfg["folds"]:
        raw = torch.load(cfg["data"](k), map_location="cpu", weights_only=False)
        pred, true, m = predict_test(cfg["run"](k) / "best_model.pth", raw,
                                     sensor_file(net), TOPOLOGY[net], dev)
        if mask is None:
            mask = m
        assert np.array_equal(mask, m), "sensor mask differs across folds"
        P.append(pred.astype(np.float32))
        Y.append(true.astype(np.float32))
        if k is not None:
            print(f"  fold {k}: {len(pred)} records, MAE {np.abs(pred - true)[:, m].mean():.2f} cm")
        del raw
        gc.collect()

    pred, true = np.concatenate(P), np.concatenate(Y)
    out = dict(preds=pred, trues=true, non_sensor=mask.astype(bool))
    out.update(per_node(pred, true))
    out.update(globals_(pred, true, mask))
    PRED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PRED / f"{net}.npz", **out)
    print(f"{net}: MAE {out['mae_global_cm']:.2f} cm over {len(pred)} records -> {PRED / net}.npz")


if __name__ == "__main__":
    for net in sys.argv[1:] or ["heusden", "bellinge", "tuindorp"]:
        build(net)
