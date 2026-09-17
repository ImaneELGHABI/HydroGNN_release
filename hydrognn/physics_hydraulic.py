"""Asset-specific hydraulic laws for urban drainage, as opt-in loss terms.

SEPARATE FROM physics_loss.py BY DESIGN. Nothing here is imported by the existing
terms and no existing behaviour changes; every term is off unless explicitly enabled.

Why this module exists
----------------------
The terms in physics_loss.py apply one proxy, Q = relu(level_us - level_ds), to
every gravity asset, and exclude pumps entirely. Two consequences:

  * It thresholds on the wrong quantity for a weir. A weir passes zero discharge
    unless the UPSTREAM head exceeds a fixed CREST elevation; the proxy instead
    triggers whenever the upstream node sits above the downstream one. Weirs sit
    high and are dry most of the time, so the proxy fabricates flow through them
    in the network's normal state -- it is not a weak weir model, it is not a
    weir model.
  * Pumps contribute no edges at all, so their switching thresholds -- which are
    present in the data -- cannot influence anything.

The laws below use parameters that were verified against the InfoWorks topology
export rather than assumed:

  weir edge_attr = [crest, width, discharge_coeff, height]
      matched by exact min/max agreement with the weir CSV columns crest, width,
      discharge_coeff (0.600 for all 95 weirs) and height.
  pump edge_attr = [discharge, switch_on_level, switch_off_level, coeff, flag]
      switch_on > switch_off for 111/111 pumps, and both ranges agree with the
      pump CSV.

UNITS AND DATUM. All laws work in physical metres on the WATER LEVEL, so two
conventions have to be handled explicitly and neither can be assumed:

  target frame  Heusden (InfoWorks depnod) predicts a level above datum, so the
                level IS the denormalised prediction. Bellinge/Tuindorp (SWMM)
                predict depth above invert, so the level is prediction + invert.
                physics_loss.compute_water_levels always adds the invert, which
                is right for SWMM and wrong for Heusden.
  elevation     the Heusden export writes elevations in feet against a metric
    scale       target. Whatever divides the invert must divide the crest too, or
                the threshold lands in the wrong place by a factor of 3.28.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

FEET_PER_METRE = 3.280839895013123
G = 9.81

# Verified column layouts (see module docstring).
WEIR_CREST, WEIR_WIDTH, WEIR_CD, WEIR_HEIGHT = 0, 1, 2, 3
PUMP_Q, PUMP_ON, PUMP_OFF = 0, 1, 2


def levels_from_pred(pred_znorm: torch.Tensor, invert_m: Optional[torch.Tensor],
                     depth_mean: float, depth_std: float,
                     target_frame: str = "level") -> torch.Tensor:
    """Denormalised water level in metres, honouring the dataset's target convention.

    target_frame='level' : prediction is already a level above datum (InfoWorks)
    target_frame='depth' : prediction is depth above invert (SWMM), so add invert
    """
    v = pred_znorm * depth_std + depth_mean
    if target_frame == "depth":
        if invert_m is None:
            raise ValueError("target_frame='depth' needs invert_m")
        return F.relu(v) + invert_m
    return v


def weir_discharge(h_us: torch.Tensor, h_ds: torch.Tensor, crest: torch.Tensor,
                   width: torch.Tensor, cd: torch.Tensor,
                   modular_limit: float = 0.8,
                   eps: float = 0.01) -> torch.Tensor:
    """Free/submerged sharp-crested weir discharge, in m^3/s.

        Q_free = Cd * (2/3) * sqrt(2g) * b * (h_us - crest)^{3/2}

    with the Villemonte submergence correction applied once the downstream head
    over crest exceeds `modular_limit` of the upstream head over crest:

        Q = Q_free * [1 - (H_ds/H_us)^{3/2}]^{0.385}

    `eps` softens the crest so the 3/2 power has a usable gradient at the
    threshold instead of a vanishing one; anneal it toward zero rather than
    annealing the loss weight, which is what actually controls how sharp the
    switch is.
    """
    head_us = F.relu(h_us - crest)
    head_ds = F.relu(h_ds - crest)
    q_free = cd * (2.0 / 3.0) * (2.0 * G) ** 0.5 * width * (head_us + eps) ** 1.5
    q_free = q_free * (head_us > 0).to(q_free.dtype)

    # Guard the denominator separately from eps: with eps=0 a dry weir gives
    # 0/0 = NaN, and 0 * NaN is NaN, so the zero-discharge case would poison the
    # whole loss rather than contributing nothing.
    ratio = head_ds / torch.clamp(head_us + eps, min=1e-9)
    submerged = (ratio > modular_limit).to(q_free.dtype)
    villemonte = torch.clamp(1.0 - torch.clamp(ratio, 0.0, 1.0) ** 1.5, min=1e-6) ** 0.385
    return q_free * (1.0 - submerged + submerged * villemonte)


def pump_discharge(h_us: torch.Tensor, on_level: torch.Tensor,
                   off_level: torch.Tensor, q_rated: torch.Tensor,
                   tau: float = 0.05) -> torch.Tensor:
    """Pump discharge under a soft switching threshold, in m^3/s.

    A real pump has hysteresis: it starts at `on_level` and stops at `off_level`,
    which is a function of history. A per-snapshot model has no history, so the
    band between the two levels is represented by a smooth ramp -- the pump is
    certainly off below `off_level`, certainly on above `on_level`, and the
    sigmoid interpolates in between rather than pretending to know the state.
    """
    mid = 0.5 * (on_level + off_level)
    band = torch.clamp(on_level - off_level, min=tau)
    # 6/band puts off_level at -3 and on_level at +3, so the pump is ~5% of rated
    # at its stop level and ~95% at its start level instead of drifting across
    # the whole range.
    return q_rated * torch.sigmoid((h_us - mid) * 6.0 / band)


def _endpoints(edge_index_dict: Dict, group: str
               ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    ei_us = edge_index_dict.get(("manhole", "has_us", group))
    ei_ds = edge_index_dict.get((group, "has_ds", "manhole"))
    if ei_us is None or ei_ds is None or ei_us.size(1) == 0:
        return None
    if ei_us.size(1) != ei_ds.size(1):
        return None
    return ei_us[0][ei_us[1].argsort()], ei_ds[1][ei_ds[0].argsort()]


def weir_consistency_loss(level_m: torch.Tensor, x_dict: Dict,
                          edge_index_dict: Dict, elev_scale: float = 1.0,
                          eps: float = 0.01) -> torch.Tensor:
    """UNSUPERVISED: a weir below its crest must not drive a downstream rise.

    Needs no target and no time step. Where the predicted upstream head is below
    the crest the weir is passing nothing, so the prediction should not place the
    downstream node above the upstream one *through that weir*. Violations are
    penalised; a dry weir contributes zero rather than the phantom discharge the
    relu(delta level) proxy invents.
    """
    ep = _endpoints(edge_index_dict, "weir")
    if ep is None or "weir" not in x_dict:
        return level_m.new_zeros(())
    us, ds = ep
    w = x_dict["weir"]
    if w.size(1) <= WEIR_HEIGHT:
        return level_m.new_zeros(())
    crest = w[:, WEIR_CREST] / elev_scale
    dry = (level_m[us] < crest).to(level_m.dtype)
    rise = F.relu(level_m[ds] - level_m[us])
    return (dry * rise).mean() if dry.numel() else level_m.new_zeros(())


def bounds_loss(level_m: torch.Tensor, invert_m: torch.Tensor,
                ground_m: Optional[torch.Tensor] = None) -> torch.Tensor:
    """UNSUPERVISED: water level may not fall below the invert, nor exceed ground.

    Needs no target. Level below invert is physically impossible; level above
    ground level is flooding, which the 1D model represents as spill rather than
    unbounded head, so it is penalised as a soft ceiling.
    """
    below = F.relu(invert_m - level_m).mean()
    if ground_m is None:
        return below
    return below + F.relu(level_m - ground_m).mean()


def asset_flow_residual(level_m: torch.Tensor, x_dict: Dict,
                        edge_index_dict: Dict, target_level_m: torch.Tensor,
                        elev_scale: float = 1.0,
                        per_class: bool = True) -> torch.Tensor:
    """Continuity residual using the correct law per asset class.

    Discharges come from the weir and pump laws above rather than from
    relu(delta level). Residuals are normalised WITHIN each asset class before
    averaging, because weirs are 1.78% of the gravity edges on Heusden and a
    global mean makes even an exact weir law invisible.

    This form still compares the predicted residual against the residual the
    target exhibits, so it remains anchored to the target: it isolates whether
    correct hydraulic laws help inside the existing objective. Making the
    constraint genuinely unsupervised needs the storage term dS/dt and the known
    lateral inflow, which requires paired snapshots and the qrain/qinfnod
    exports -- see the module docstring in the follow-up work.
    """
    terms = []
    for group, fn in (("weir", "weir"), ("pump", "pump")):
        ep = _endpoints(edge_index_dict, group)
        if ep is None or group not in x_dict:
            continue
        us, ds = ep
        a = x_dict[group]
        if group == "weir":
            if a.size(1) <= WEIR_HEIGHT:
                continue
            crest = a[:, WEIR_CREST] / elev_scale
            args = (crest, a[:, WEIR_WIDTH], a[:, WEIR_CD])
            qp = weir_discharge(level_m[us], level_m[ds], *args)
            qt = weir_discharge(target_level_m[us], target_level_m[ds], *args)
        else:
            if a.size(1) <= PUMP_OFF:
                continue
            on, off = a[:, PUMP_ON] / elev_scale, a[:, PUMP_OFF] / elev_scale
            qr = a[:, PUMP_Q]
            qp = pump_discharge(level_m[us], on, off, qr)
            qt = pump_discharge(target_level_m[us], on, off, qr)
        d = (qp - qt).abs()
        terms.append(d.mean() / (qt.abs().mean() + 1e-6) if per_class else d.mean())
    if not terms:
        return level_m.new_zeros(())
    return torch.stack(terms).mean()


def hydraulic_physics_loss(pred_znorm: torch.Tensor, target_znorm: torch.Tensor,
                           invert_m: Optional[torch.Tensor], x_dict: Dict,
                           edge_index_dict: Dict, depth_mean: float,
                           depth_std: float, target_frame: str = "level",
                           elev_scale: float = 1.0,
                           ground_m: Optional[torch.Tensor] = None,
                           lambda_asset: float = 0.0,
                           lambda_weir_dry: float = 0.0,
                           lambda_bounds: float = 0.0
                           ) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Combine the enabled hydraulic terms. Returns (loss, components)."""
    level_p = levels_from_pred(pred_znorm, invert_m, depth_mean, depth_std, target_frame)
    total = pred_znorm.new_zeros(())
    comps: Dict[str, float] = {}

    if lambda_asset > 0.0:
        level_t = levels_from_pred(target_znorm, invert_m, depth_mean, depth_std,
                                   target_frame)
        l = asset_flow_residual(level_p, x_dict, edge_index_dict, level_t, elev_scale)
        total = total + lambda_asset * l
        comps["L_asset"] = float(l.detach())
    if lambda_weir_dry > 0.0:
        l = weir_consistency_loss(level_p, x_dict, edge_index_dict, elev_scale)
        total = total + lambda_weir_dry * l
        comps["L_weirdry"] = float(l.detach())
    if lambda_bounds > 0.0 and invert_m is not None:
        l = bounds_loss(level_p, invert_m, ground_m)
        total = total + lambda_bounds * l
        comps["L_bounds"] = float(l.detach())
    return total, comps
