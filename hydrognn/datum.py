"""Elevation datum reconciliation between the target and the invert attribute.

On Heusden the target is the simulated water level in metres above datum
(InfoWorks ICM `depnod`) while `manhole.invert_level_m` is stored on a different
datum: `level - invert` is negative for 83.9% of nodes, with a minimum of
-13.39 m. A depth cannot be negative, so any physics term, dry-node mask or
depth reparameterisation built on that subtraction is operating on nonsense.

The two quantities are nonetheless almost perfectly collinear across nodes
(r = 0.996), so the offset is a datum shift rather than missing information, and
it can be recovered by fitting from the training split alone.

Three modes:

  check   measure the disagreement and raise if it exceeds a tolerance. Use this
          as a guard in preprocessing so a mismatched dataset fails loudly
          instead of silently training on impossible depths.
  align   keep the level target and rewrite invert_level_m onto the target datum
          via a scale/offset fit on the TRAIN graphs only, so that
          level - invert is a physically meaningful depth.
  depth   convert the target itself to depth above invert, clamped at zero, and
          record the per-node offset needed to reconstruct absolute level.
  patch_units
          divide invert by a fixed factor (default 3.28084, feet to metres) on
          the hypothesis that the attribute was exported in imperial units, then
          re-run the check. Diagnostic only: see the warning in
          `patch_invert_units` about why this cannot change training.

`align` and `depth` both fit on training snapshots only; validation and test
graphs are transformed with the fitted constants and never contribute to them.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch


def _levels(graphs: List, depth_mean: float, depth_std: float) -> torch.Tensor:
    """[T, N] target levels in metres."""
    ys = []
    for g in graphs:
        y = g["manhole"].y
        ys.append((y[:, 0] if y.dim() == 2 else y).float())
    return torch.stack(ys) * depth_std + depth_mean


def _invert(graphs: List) -> Optional[torch.Tensor]:
    iv = getattr(graphs[0]["manhole"], "invert_level_m", None)
    return None if iv is None else iv.float()


def datum_report(graphs: List, depth_mean: float, depth_std: float) -> dict:
    """Disagreement between the target datum and the invert datum."""
    iv = _invert(graphs)
    if iv is None:
        return {"has_invert": False}
    lv = _levels(graphs, depth_mean, depth_std)
    d = lv - iv.unsqueeze(0)
    node_mean_level = lv.mean(0)
    r = float(torch.corrcoef(torch.stack([node_mean_level, iv]))[0, 1])
    return {"has_invert": True,
            "frac_negative_depth": float((d < 0).float().mean()),
            "min_depth_m": float(d.min()),
            "median_depth_m": float(d.median()),
            "corr_level_invert": r}


def fit_invert_alignment(train_graphs: List, depth_mean: float,
                         depth_std: float) -> Tuple[float, float]:
    """Least-squares (a, b) mapping invert onto the target datum, train only.

    Returns coefficients such that `a * invert + b` sits on the same datum as the
    level target. Fitted against each node's mean level over the training
    snapshots, which removes storm-to-storm variation and leaves the static
    elevation relationship.
    """
    iv = _invert(train_graphs)
    if iv is None:
        raise ValueError("graphs carry no invert_level_m; cannot align datums")
    y = _levels(train_graphs, depth_mean, depth_std).mean(0)
    A = torch.stack([iv, torch.ones_like(iv)], dim=1)
    sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution.squeeze(1)
    return float(sol[0]), float(sol[1])


FEET_PER_METRE = 3.280839895013123


def patch_invert_units(graph_sets: List[List], factor: float = FEET_PER_METRE) -> None:
    """Divide invert_level_m by `factor` in place, across every supplied set.

    Heusden's invert attribute is in feet. The model input column is z-scored,
    so this only affects quantities computed from level - invert (the auxiliary
    loss terms).
    """
    for gs in graph_sets:
        for g in gs:
            iv = getattr(g["manhole"], "invert_level_m", None)
            if iv is not None:
                g["manhole"].invert_level_m = iv.float() / factor


def enforce_datum(train_graphs: List, other_graph_sets: List[List],
                  depth_mean: float, depth_std: float,
                  mode: str = "check", max_frac_negative: float = 0.02,
                  unit_factor: float = FEET_PER_METRE, verbose: bool = True):
    """Guard or reconcile the datum. Returns a dict describing what was done.

    mode='check'  raise if more than `max_frac_negative` of level-invert values
                  are negative.
    mode='align'  rewrite invert_level_m onto the target datum (train-fitted).
    mode='depth'  replace the target with depth above invert, clamped at 0, and
                  return the per-node offset needed to rebuild absolute level.
    mode='patch_units'
                  divide invert by `unit_factor`, then re-check.
    """
    rep = datum_report(train_graphs, depth_mean, depth_std)
    if not rep["has_invert"]:
        if mode != "check":
            raise ValueError("no invert_level_m present; cannot apply mode=" + mode)
        if verbose:
            print("  Datum guard: no invert_level_m on this dataset, nothing to check.")
        return {"mode": mode, **rep}
    if verbose:
        print(f"  Datum guard: level-invert negative for "
              f"{rep['frac_negative_depth']:.1%} of node-snapshots "
              f"(min {rep['min_depth_m']:.2f} m, "
              f"corr(level, invert) = {rep['corr_level_invert']:+.3f})")

    if mode == "patch_units":
        patch_invert_units([train_graphs] + list(other_graph_sets), unit_factor)
        after = datum_report(train_graphs, depth_mean, depth_std)
        if verbose:
            print(f"  Datum guard: invert divided by {unit_factor:.5f}; negative "
                  f"{rep['frac_negative_depth']:.1%} -> "
                  f"{after['frac_negative_depth']:.1%}, median depth "
                  f"{after['median_depth_m']:+.3f} m")
        return {"mode": mode, "factor": unit_factor, "before": rep, "after": after}

    if mode == "check":
        if rep["frac_negative_depth"] > max_frac_negative:
            raise AssertionError(
                f"target and invert_level_m are on inconsistent datums: "
                f"{rep['frac_negative_depth']:.1%} of level-invert values are "
                f"negative (min {rep['min_depth_m']:.2f} m), which is not a "
                f"physical depth. corr(level, invert) = "
                f"{rep['corr_level_invert']:+.3f} indicates a datum shift rather "
                f"than missing information: re-run with mode='align' or 'depth', "
                f"or export invert on the target's datum from InfoWorks ICM.")
        return {"mode": mode, **rep}

    a, b = fit_invert_alignment(train_graphs, depth_mean, depth_std)
    if verbose:
        print(f"  Datum guard: fitted invert -> target datum as "
              f"{a:.4f} * invert + {b:.4f} (train split only)")

    if mode == "align":
        for gs in [train_graphs] + list(other_graph_sets):
            for g in gs:
                iv = g["manhole"].invert_level_m.float()
                g["manhole"].invert_level_m = a * iv + b
        after = datum_report(train_graphs, depth_mean, depth_std)
        if verbose:
            print(f"  Datum guard: after alignment "
                  f"{after['frac_negative_depth']:.1%} negative, "
                  f"median depth {after['median_depth_m']:.2f} m")
        return {"mode": mode, "a": a, "b": b, "before": rep, "after": after}

    if mode == "depth":
        # target <- clamp(level - (a*invert+b), 0); offset rebuilds the level
        offset_m = None
        for gs in [train_graphs] + list(other_graph_sets):
            for g in gs:
                iv = g["manhole"].invert_level_m.float()
                base = a * iv + b
                y = g["manhole"].y
                flat = (y[:, 0] if y.dim() == 2 else y).float()
                lvl = flat * depth_std + depth_mean
                dep = torch.clamp(lvl - base, min=0.0)
                new = (dep - depth_mean) / depth_std
                if y.dim() == 2:
                    y[:, 0] = new
                else:
                    g["manhole"].y = new
                g["manhole"].level_offset_m = base
                offset_m = base
        if verbose:
            print("  Datum guard: target converted to depth above invert, "
                  "clamped at 0; level_offset_m recorded per node")
        return {"mode": mode, "a": a, "b": b, "offset_m": offset_m, "before": rep}

    raise ValueError(f"unknown mode {mode!r}")
