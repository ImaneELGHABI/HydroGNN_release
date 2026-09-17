"""HydroGNN physics-informed loss.

    L_total = λ_pred·L_pred + λ_flow·L_flow + λ_qmax·L_qmax + λ_depth·L_depth + λ_level·L_level

    L_flow  : Σ |inflow − outflow|  for each manhole          (mass conservation)
    L_qmax  : Σ max(0, Q_pred − Q_max)                        (Manning pipe capacity)
    L_depth : Σ max(0, h_upstream − h_downstream)             (depth monotonicity / gravity)
    L_level : Σ max(0, level_downstream − level_upstream)     (water surface monotonicity)

Two graph representations are supported and detected automatically:
  - Assets as edges: ("manhole", "conduit", "manhole"), ("manhole", "weir", "manhole"), ...
  - Assets as nodes: ("manhole", "has_us", "pipe"), ("pipe", "has_ds", "manhole"), ...

Pumps are excluded from gravity-based terms (L_qmax, L_depth, L_level).
"""
from __future__ import annotations
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── learnable task weighting ─────────────────────────────────────────────────
class LearnableLossWeights(nn.Module):
    r"""Homoscedastic task-uncertainty weighting (Kendall et al., 2018).

    Replaces hand-set :math:`\lambda` coefficients with learned task variances:

    .. math::
        \mathcal{L} = \sum_t \frac{1}{2\sigma_t^2}\mathcal{L}_t
                    + \sum_t \log \sigma_t

    Parameterised by :math:`\log \sigma_t^2` for unconstrained optimisation, so
    :math:`1/(2\sigma_t^2) = \tfrac12 e^{-\log\sigma_t^2}` and
    :math:`\log\sigma_t = \tfrac12 \log\sigma_t^2`. The log term is what stops
    the trivial solution of driving every precision to zero.

    A task's precision is only meaningful while that task contributes to the
    objective, so `regulariser()` takes the active subset: during a physics
    warmup only `data` is active and only its log-variance is penalised.

    These parameters must be handed to the optimiser (see train.py) or they stay
    frozen at their initial values and the weighting silently does nothing.
    """

    def __init__(self, tasks: Tuple[str, ...] = ("data", "mass", "grad"),
                 init_log_var: float = 0.0):
        super().__init__()
        self.tasks = tuple(tasks)
        self.log_var = nn.Parameter(torch.full((len(self.tasks),), float(init_log_var)))

    def _idx(self, task: str) -> int:
        if task not in self.tasks:
            raise KeyError(f"unknown task {task!r}; have {self.tasks}")
        return self.tasks.index(task)

    def precision(self, task: str) -> torch.Tensor:
        """1 / (2 sigma^2) for the given task."""
        return 0.5 * torch.exp(-self.log_var[self._idx(task)])

    def regulariser(self, tasks: Optional[Tuple[str, ...]] = None) -> torch.Tensor:
        """sum_t log sigma_t over the active tasks."""
        idx = [self._idx(t) for t in (tasks if tasks is not None else self.tasks)]
        return 0.5 * self.log_var[torch.tensor(idx, device=self.log_var.device)].sum()

    def sigmas(self) -> Dict[str, float]:
        with torch.no_grad():
            return {t: float(torch.exp(0.5 * self.log_var[i]))
                    for i, t in enumerate(self.tasks)}


# ── format detection ─────────────────────────────────────────────────────────

_EDGE_GRAVITY_RELS = {"conduit", "channel", "weir", "orifice", "flap_valve"}

def _has_asset_nodes(edge_index_dict: dict) -> bool:
    """True when assets are dedicated nodes (lines-as-nodes format)."""
    return ("manhole", "has_us", "pipe") in edge_index_dict


# ── shared: convert z-norm → physical units ───────────────────────────────────

def compute_water_levels(
    pred_znorm: torch.Tensor,
    invert_level_m: torch.Tensor,
    depth_mean: float,
    depth_std: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (depth_m, water_level_m) in physical units."""
    depth_m = F.relu(pred_znorm * depth_std + depth_mean)
    return depth_m, depth_m + invert_level_m


# ── shared: gravity-asset endpoint pairs ─────────────────────────────────────

def _gravity_endpoints(
    edge_index_dict: dict,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (us_manhole_idx, ds_manhole_idx) for every gravity asset (no pumps).

    Works for both graph representations: assets as edges and assets as dedicated nodes.
    """
    if _has_asset_nodes(edge_index_dict):
        us_list, ds_list = [], []
        for group in ("pipe", "weir"):
            ei_us = edge_index_dict.get(("manhole", "has_us", group))
            ei_ds = edge_index_dict.get((group, "has_ds", "manhole"))
            if ei_us is None or ei_ds is None or ei_us.size(1) == 0:
                continue
            # Sort by asset-node index so the two arrays align per asset
            ord_us = ei_us[1].argsort()
            ord_ds = ei_ds[0].argsort()
            us_list.append(ei_us[0][ord_us])
            ds_list.append(ei_ds[1][ord_ds])
        if not us_list:
            return None, None
        return torch.cat(us_list), torch.cat(ds_list)
    else:
        us_list, ds_list = [], []
        for (src, rel, dst), ei in edge_index_dict.items():
            if src == "manhole" and dst == "manhole" and rel in _EDGE_GRAVITY_RELS:
                if ei.size(1) > 0:
                    us_list.append(ei[0])
                    ds_list.append(ei[1])
        if not us_list:
            return None, None
        return torch.cat(us_list), torch.cat(ds_list)


def _pipe_endpoints_and_features(
    edge_index_dict: dict,
    edge_attr_dict: Optional[dict],
    node_x_dict: Optional[dict],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return (us_mh, ds_mh, pipe_attr) for conduit pipes only (used by L_qmax).

    pipe_attr columns (both formats): [diameter_m, length/100, manning_n, ...]
    """
    if _has_asset_nodes(edge_index_dict):
        ei_us = edge_index_dict.get(("manhole", "has_us", "pipe"))
        ei_ds = edge_index_dict.get(("pipe", "has_ds", "manhole"))
        if ei_us is None or ei_ds is None or ei_us.size(1) == 0:
            return None, None, None
        if node_x_dict is None or "pipe" not in node_x_dict:
            return None, None, None
        ord_us = ei_us[1].argsort()
        ord_ds = ei_ds[0].argsort()
        us_mh = ei_us[0][ord_us]
        ds_mh = ei_ds[1][ord_ds]
        pipe_attr = node_x_dict["pipe"][ord_us]   # [n_pipes, D_pipe]; cols 0-3 = D,L/100,n,slope
        return us_mh, ds_mh, pipe_attr
    else:
        key = ("manhole", "conduit", "manhole")
        ei = edge_index_dict.get(key)
        ea = (edge_attr_dict or {}).get(key)
        if ei is None or ei.size(1) == 0 or ea is None:
            return None, None, None
        return ei[0], ei[1], ea


# ── L_flow ────────────────────────────────────────────────────────────────────

def flow_conservation_loss(
    water_level_m: torch.Tensor,
    edge_index_dict: dict,
) -> torch.Tensor:
    """L_flow = mean( |inflow − outflow| ) for each manhole.

    Flow proxy: Q = max(0, level_us − level_ds).
    Excludes pumps (active control, not gravity-driven).
    """
    N = water_level_m.size(0)
    net = water_level_m.new_zeros(N)
    us, ds = _gravity_endpoints(edge_index_dict)
    if us is None:
        return water_level_m.new_zeros(())

    flow = F.relu(water_level_m[us] - water_level_m[ds])
    net.scatter_add_(0, ds, flow)
    net.scatter_add_(0, us, -flow)
    return net.abs().mean()


# ── L_qmax ────────────────────────────────────────────────────────────────────

def qmax_loss(
    water_level_m: torch.Tensor,
    invert_level_m: torch.Tensor,
    edge_index_dict: dict,
    edge_attr_dict: Optional[dict],
    node_x_dict: Optional[dict] = None,
    min_slope: float = 1e-5,
) -> torch.Tensor:
    """L_qmax = mean( max(0, Q_predicted − Q_max) ) using Manning's equation.

    pipe_attr cols: [diameter_m, length/100, manning_n, ...]
    """
    us, ds, pa = _pipe_endpoints_and_features(edge_index_dict, edge_attr_dict, node_x_dict)
    if us is None or pa is None:
        return water_level_m.new_zeros(())

    D = pa[:, 0].clamp(min=0.05)
    L = (pa[:, 1] * 100.0).clamp(min=0.1)
    n = pa[:, 2].clamp(min=0.001)

    A = math.pi * (D / 2.0) ** 2
    R = D / 4.0

    S_struct = ((invert_level_m[us] - invert_level_m[ds]) / L).clamp(min=min_slope)
    Q_max = (1.0 / n) * A * R.pow(2.0 / 3.0) * S_struct.pow(0.5)

    delta_level = F.relu(water_level_m[us] - water_level_m[ds])
    S_pred = (delta_level / L).clamp(min=min_slope)
    Q_pred = (1.0 / n) * A * R.pow(2.0 / 3.0) * S_pred.pow(0.5)

    return F.relu(Q_pred - Q_max).mean()


# ── L_depth ───────────────────────────────────────────────────────────────────

def depth_monotonicity_loss(
    depth_m: torch.Tensor,
    edge_index_dict: dict,
) -> torch.Tensor:
    """L_depth = mean( max(0, h_upstream − h_downstream) ).

    Gravity: depth increases going downstream (downstream manhole fills deeper).
    """
    us, ds = _gravity_endpoints(edge_index_dict)
    if us is None:
        return depth_m.new_zeros(())
    return F.relu(depth_m[us] - depth_m[ds]).mean()


# ── L_level ───────────────────────────────────────────────────────────────────

def water_level_monotonicity_loss(
    water_level_m: torch.Tensor,
    edge_index_dict: dict,
) -> torch.Tensor:
    """L_level = mean( max(0, level_downstream − level_upstream) ).

    Water surface must decrease going downstream (gravity flow).
    """
    us, ds = _gravity_endpoints(edge_index_dict)
    if us is None:
        return water_level_m.new_zeros(())
    return F.relu(water_level_m[ds] - water_level_m[us]).mean()


# ── L_grad ───────────────────────────────────────────────────────────────────

def gradient_matching_loss(
    pred_znorm: torch.Tensor,
    target_znorm: torch.Tensor,
    edge_index_dict: dict,
    min_grad_thresh: float = 1e-4,
) -> torch.Tensor:
    """L_grad = MSE( (pred[ds]-pred[us]) - (target[ds]-target[us]) ) along gravity edges.

    Penalises incorrect spatial depth gradients between connected manholes.
    Only edges with a meaningful true gradient (|Δtarget| >= min_grad_thresh) are used.
    """
    us, ds = _gravity_endpoints(edge_index_dict)
    if us is None:
        return pred_znorm.new_zeros(())
    true_grad = target_znorm[ds] - target_znorm[us]
    pred_grad = pred_znorm[ds] - pred_znorm[us]
    mask = true_grad.abs() >= min_grad_thresh
    if not mask.any():
        return pred_znorm.new_zeros(())
    diff = pred_grad[mask] - true_grad[mask]
    return (diff ** 2).mean()


# ── L_mass ────────────────────────────────────────────────────────────────────

def mass_conservation_loss_znorm(
    pred_znorm: torch.Tensor,
    edge_index_dict: dict,
) -> torch.Tensor:
    """L_mass = mean |net_flow| per manhole, computed in z-norm space.

    Flow proxy: Q = relu(pred[us] - pred[ds]).  Operating in z-norm (not physical
    units) lets the loss scale with prediction errors and decrease as training
    converges, unlike the Manning-based L_flow which stays high due to invert offsets.
    """
    N = pred_znorm.size(0)
    net = pred_znorm.new_zeros(N)
    us, ds = _gravity_endpoints(edge_index_dict)
    if us is None:
        return pred_znorm.new_zeros(())
    flow = F.relu(pred_znorm[us] - pred_znorm[ds])
    net.scatter_add_(0, ds, flow)
    net.scatter_add_(0, us, -flow)
    return net.abs().mean()


def mass_conservation_loss_corrected(
    pred_znorm: torch.Tensor,
    target_znorm: torch.Tensor,
    invert_level_m: torch.Tensor,
    edge_index_dict: dict,
    depth_mean: float,
    depth_std: float,
) -> torch.Tensor:
    """Residual-matching mass balance (--corrected_mass).

    Flow direction follows the water surface (invert + depth). The prediction's
    nodal residual is matched to the residual of the ground truth rather than to
    zero: manholes fill and drain during a storm, and a single snapshot carries
    no dV/dt. A spatially flat prediction gives zero net flow and is penalised.

    Returned in z-norm units so its magnitude stays comparable to L_pred.
    """
    us, ds = _gravity_endpoints(edge_index_dict)
    if us is None:
        return pred_znorm.new_zeros(())
    if invert_level_m is None:
        invert_level_m = pred_znorm.new_zeros(pred_znorm.size(0))

    _, level_pred = compute_water_levels(pred_znorm, invert_level_m, depth_mean, depth_std)
    _, level_true = compute_water_levels(target_znorm, invert_level_m, depth_mean, depth_std)

    N = pred_znorm.size(0)
    net_p = pred_znorm.new_zeros(N)
    net_t = pred_znorm.new_zeros(N)
    flow_p = F.relu(level_pred[us] - level_pred[ds])
    flow_t = F.relu(level_true[us] - level_true[ds])
    net_p.scatter_add_(0, ds, flow_p); net_p.scatter_add_(0, us, -flow_p)
    net_t.scatter_add_(0, ds, flow_t); net_t.scatter_add_(0, us, -flow_t)
    return ((net_p - net_t).abs() / max(depth_std, 1e-6)).mean()


# ── combined ──────────────────────────────────────────────────────────────────

def physics_loss(
    pred_znorm: torch.Tensor,
    target_znorm: torch.Tensor,
    invert_level_m: torch.Tensor,
    edge_index_dict: dict,
    edge_attr_dict: Optional[dict],
    depth_mean: float,
    depth_std: float,
    *,
    mask: Optional[torch.Tensor] = None,
    node_x_dict: Optional[dict] = None,
    pred_loss: str = "mse",
    node_weights: Optional[torch.Tensor] = None,
    lambda_pred: float = 1.0,
    lambda_flow: float = 0.1,
    lambda_qmax: float = 0.01,
    lambda_depth: float = 0.1,
    lambda_level: float = 0.1,
    lambda_grad: Optional[float] = None,
    lambda_mass: Optional[float] = None,
    loss_weights: Optional[LearnableLossWeights] = None,
    physics_active: bool = True,
    corrected_mass: bool = False,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """L_total = λ_pred·L_pred + λ_flow·L_flow + λ_qmax·L_qmax + λ_depth·L_depth + λ_level·L_level

    Args:
        pred_znorm     : [N_manhole] model output, z-normalised
        target_znorm   : [N_manhole] ground truth, z-normalised
        invert_level_m : [N_manhole] chamber floor elevation [m AD]
        edge_index_dict: batched heterogeneous edge indices
        edge_attr_dict : batched edge attributes (used when assets are edges; pass None otherwise)
        depth_mean/std : z-norm parameters
        mask           : [N_manhole] bool — True = non-sensor node (used for L_pred)
        node_x_dict    : x_dict from the batch (needed for L_qmax when assets are dedicated nodes)
        pred_loss      : "mse" or "mae" — error metric for L_pred
        node_weights   : [N_manhole] per-node L_pred weights (e.g. by system type)
        lambda_*       : loss weights
    """
    sel = mask if (mask is not None and mask.sum() > 0) \
          else torch.ones_like(pred_znorm, dtype=torch.bool)
    err = pred_znorm[sel] - target_znorm[sel]
    per_node = err.abs() if pred_loss == "mae" else err ** 2
    if node_weights is not None:
        w = node_weights[sel]
        l_pred = (per_node * w).sum() / w.sum().clamp(min=1e-8)
    else:
        l_pred = per_node.mean()

    if invert_level_m is None:
        invert_level_m = pred_znorm.new_zeros(pred_znorm.size(0))

    zero = pred_znorm.new_zeros(())
    if lambda_flow > 0 or lambda_qmax > 0 or lambda_depth > 0 or lambda_level > 0:
        depth_m, water_level_m = compute_water_levels(pred_znorm, invert_level_m, depth_mean, depth_std)
        l_flow  = flow_conservation_loss(water_level_m, edge_index_dict) if lambda_flow > 0 else zero
        l_qmax  = qmax_loss(water_level_m, invert_level_m, edge_index_dict, edge_attr_dict, node_x_dict) \
                  if lambda_qmax > 0 else zero
        l_depth = depth_monotonicity_loss(depth_m, edge_index_dict) if lambda_depth > 0 else zero
        l_level = water_level_monotonicity_loss(water_level_m, edge_index_dict) if lambda_level > 0 else zero
    else:
        l_flow = l_qmax = l_depth = l_level = zero

    # The auxiliary gravity terms keep their static weights; only the data,
    # mass and gradient terms take part in the learned weighting, matching the
    # three-task formulation.
    total = (lambda_flow  * l_flow
           + lambda_qmax  * l_qmax
           + lambda_depth * l_depth
           + lambda_level * l_level)

    comps = {
        "L_pred":  float(l_pred.detach()),
        "L_flow":  float(l_flow.detach()),
        "L_qmax":  float(l_qmax.detach()),
        "L_depth": float(l_depth.detach()),
        "L_level": float(l_level.detach()),
    }

    if loss_weights is None:
        total = total + lambda_pred * l_pred
        if lambda_grad is not None and lambda_grad > 0.0:
            l_grad = gradient_matching_loss(pred_znorm, target_znorm, edge_index_dict)
            total = total + lambda_grad * l_grad
            comps["L_grad"] = float(l_grad.detach())
        if lambda_mass is not None and lambda_mass > 0.0:
            l_mass = (mass_conservation_loss_corrected(
                          pred_znorm, target_znorm, invert_level_m,
                          edge_index_dict, depth_mean, depth_std)
                      if corrected_mass else
                      mass_conservation_loss_znorm(pred_znorm, edge_index_dict))
            total = total + lambda_mass * l_mass
            comps["L_mass"] = float(l_mass.detach())
        return total, comps

    # ── learned task weighting ───────────────────────────────────────────────
    active = ["data"]
    total = total + loss_weights.precision("data") * l_pred

    if physics_active:
        l_grad = gradient_matching_loss(pred_znorm, target_znorm, edge_index_dict)
        l_mass = (mass_conservation_loss_corrected(
                      pred_znorm, target_znorm, invert_level_m,
                      edge_index_dict, depth_mean, depth_std)
                  if corrected_mass else
                  mass_conservation_loss_znorm(pred_znorm, edge_index_dict))
        total = total + loss_weights.precision("grad") * l_grad
        total = total + loss_weights.precision("mass") * l_mass
        comps["L_grad"] = float(l_grad.detach())
        comps["L_mass"] = float(l_mass.detach())
        active += ["mass", "grad"]

    total = total + loss_weights.regulariser(tuple(active))
    for task, sigma in loss_weights.sigmas().items():
        comps[f"sigma_{task}"] = sigma
    return total, comps
