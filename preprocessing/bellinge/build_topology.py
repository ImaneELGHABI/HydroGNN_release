#!/usr/bin/env python3
"""
Parse the Bellinge SWMM .inp into Heusden-shaped topology CSVs.

Outputs (in <out_dir>/):
    BellingeSWMM_node.csv      — JUNCTIONS + OUTFALLS + STORAGE  (mirrors *_node.csv)
    BellingeSWMM_conduit.csv   — CONDUITS                         (mirrors *_conduit.csv)
    BellingeSWMM_pump.csv      — PUMPS                            (mirrors *_pump.csv)
    BellingeSWMM_weir.csv      — WEIRS                            (mirrors *_weir.csv)
    BellingeSWMM_orifice.csv   — ORIFICES                         (mirrors *_orifice.csv)

Column names match the Heusden CSVs so a Heusden-style loader can be ported.
SWMM has fewer fields per element than the InfoWorks ICM export Heusden uses,
so unknown columns are written empty.
"""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path

# ---------------------------------------------------------------- SWMM parser
def parse_inp(path: Path) -> dict[str, list[list[str]]]:
    """Return {section_name: [[tok, tok, ...], ...]} (comments stripped)."""
    sections: dict[str, list[list[str]]] = {}
    cur = None
    for raw in path.read_text(encoding="latin1").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("["):
            cur = line.strip("[]").strip()
            sections.setdefault(cur, [])
            continue
        if line.startswith(";") or cur is None:
            continue
        sections[cur].append(line.split())
    return sections


# ---------------------------------------------------------------- helpers
def safe_get(row: list[str], i: int, default: str = "") -> str:
    return row[i] if i < len(row) else default


def safe_float(s: str, default: float = 0.0) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"  wrote {path.name:<32} {len(rows):>6} rows  {len(header):>3} cols")


# ---------------------------------------------------------------- node columns
NODE_COLS = [
    "ObjectTable", "node_id", "storage_array", "node_type", "asset_id",
    "system_type", "connection_type", "2d_connect_line", "lateral_node_id",
    "lateral_link_suffix", "asset_uid", "infonet_id", "x", "y",
    "ground_level", "flood_level", "shaft_area_additional", "shaft_area_add_comp",
    "shaft_area_add_simplify", "shaft_area_add_ncorrect", "shaft_area_additional_total",
    "chamber_area_additional", "chamber_area_add_comp", "chamber_area_add_simplify",
    "chamber_area_add_ncorrect", "chamber_area_additional_total",
    "chamber_roof", "chamber_floor", "chamber_area", "shaft_area",
    "flood_type", "element_area_factor_2d", "flooding_discharge_coeff",
    "benching_method", "2d_link_type", "floodable_area",
    "flood_depth_1", "flood_depth_2", "flood_area_1", "flood_area_2",
    "base_area", "perimeter", "infiltration_coeff", "porosity",
    "vegetation_level", "liner_level", "infiltratn_coeff_abv_vegn",
    "infiltratn_coeff_abv_liner", "infiltratn_coeff_blw_liner",
    "relative_stages", "inlet_input_type", "inlet_type",
    "cross_slope", "grate_width", "grate_length", "opening_length",
    "opening_height", "gutter_depression", "lateral_depression",
    "velocity_splashover", "debris", "depth_weir", "clear_opening",
    "head_discharge_id", "flow_efficiency_id", "inlet_UE_a", "inlet_UE_b",
    "n_gullies", "num_transverse_bars", "num_longitudinal_bars",
    "num_diagonal_bars", "min_area_inc_voids", "area_of_voids",
    "half_road_width", "notes", "hyperlinks",
]

CONDUIT_COLS = [
    "ObjectTable", "us_node_id", "link_suffix", "ds_node_id", "link_type",
    "asset_id", "sewer_reference", "system_type", "branch_id", "point_array",
    "is_merged", "asset_uid", "infonet_us_node_id", "infonet_ds_node_id",
    "infonet_link_suffix", "us_settlement_eff", "ds_settlement_eff",
    "solution_model", "min_computational_nodes", "critical_sewer_category",
    "taking_off_reference", "conduit_material", "design_group",
    "site_condition", "ground_condition", "conduit_type",
    "min_space_step", "slot_width", "connection_coefficient",
    "shape", "conduit_width", "conduit_height", "springing_height",
    "sediment_depth", "number_of_barrels", "roughness_type",
    "bottom_roughness_CW", "top_roughness_CW",
    "bottom_roughness_Manning", "top_roughness_Manning",
    "bottom_roughness_N", "top_roughness_N",
    "bottom_roughness_HW", "top_roughness_HW",
    "conduit_length", "inflow", "gradient", "capacity",
    "us_invert", "ds_invert", "us_headloss_type", "ds_headloss_type",
    "us_headloss_coeff", "ds_headloss_coeff", "base_height",
    "infiltration_coeff_base", "infiltration_coeff_side",
    "fill_material_conductivity", "porosity",
    "diff1d_type", "diff1d_d0", "diff1d_d1", "diff1d_d2",
    "inlet_type_code", "reverse_flow_model",
]

PUMP_COLS = [
    "ObjectTable", "us_node_id", "link_suffix", "ds_node_id", "link_type",
    "system_type", "asset_id", "sewer_reference", "point_array",
    "switch_on_level", "switch_off_level", "delay", "off_delay",
    "discharge", "base_level", "head_discharge_id",
    "minimum_flow", "maximum_flow",
    "positive_change_in_flow", "negative_change_in_flow", "threshold",
    "maximum_speed", "minimum_speed",
    "positive_change_in_speed", "negative_change_in_speed",
    "nominal_speed", "threshold_speed",
    "ds_settlement_eff", "us_settlement_eff", "nominal_flow",
    "electric_hydraulic_ratio", "branch_id",
]

WEIR_COLS = [
    "ObjectTable", "us_node_id", "link_suffix", "ds_node_id", "link_type",
    "system_type", "asset_id", "sewer_reference", "point_array",
    "crest", "width", "height", "gate_height", "length",
    "orientation", "discharge_coeff", "reverse_gate_discharge_coeff",
    "secondary_discharge_coeff", "modular_limit",
    "notch_height", "notch_angle", "notch_width", "number_of_notches",
    "ds_settlement_eff", "us_settlement_eff",
    "minimum_value", "maximum_value", "minimum_crest", "maximum_crest",
    "minimum_opening", "maximum_opening", "initial_opening",
    "positive_speed", "negative_speed", "threshold", "branch_id",
]

ORIFICE_COLS = [
    "ObjectTable", "us_node_id", "link_suffix", "ds_node_id", "link_type",
    "system_type", "asset_id", "sewer_reference", "point_array",
    "invert", "diameter", "discharge_coeff", "secondary_discharge_coeff",
    "opening_type", "limiting_discharge",
    "minimum_flow", "maximum_flow",
    "positive_change_in_flow", "negative_change_in_flow", "threshold",
    "ds_settlement_eff", "us_settlement_eff", "branch_id",
]


# ---------------------------------------------------------------- main build
def main(inp_path: Path, out_dir: Path) -> None:
    print(f"Parsing {inp_path}")
    sec = parse_inp(inp_path)

    coords = {r[0]: (float(r[1]), float(r[2])) for r in sec.get("COORDINATES", [])}
    xsect: dict[str, list[str]] = {r[0]: r[1:] for r in sec.get("XSECTIONS", [])}
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------ NODES ----------------------------------------------------
    node_rows: list[dict] = []
    # JUNCTIONS: Name Elevation MaxDepth InitDepth SurDepth Aponded
    for r in sec.get("JUNCTIONS", []):
        nid = r[0]
        elev = float(safe_get(r, 1, "0") or 0)
        max_depth = float(safe_get(r, 2, "0") or 0)
        x, y = coords.get(nid, ("", ""))
        node_rows.append({
            "ObjectTable": "hw_node", "node_id": nid, "node_type": "Manhole",
            "asset_id": "Inspectieput", "system_type": "combined",
            "connection_type": "Lost", "x": x, "y": y,
            "chamber_floor": elev, "chamber_roof": elev + max_depth,
            "ground_level": elev + max_depth, "flood_level": elev + max_depth,
            "flood_type": "Stored",
        })
    # OUTFALLS: Name Elevation Type StageData Gated RouteTo
    for r in sec.get("OUTFALLS", []):
        nid = r[0]
        elev = float(safe_get(r, 1, "0") or 0)
        x, y = coords.get(nid, ("", ""))
        node_rows.append({
            "ObjectTable": "hw_node", "node_id": nid, "node_type": "Outfall",
            "asset_id": "Outfall", "system_type": "combined", "x": x, "y": y,
            "chamber_floor": elev, "chamber_roof": elev,
            "ground_level": elev, "flood_level": elev,
            "flood_type": "Lost",
        })
    # STORAGE: Name Elev MaxDepth InitDepth Shape ...
    for r in sec.get("STORAGE", []):
        nid = r[0]
        elev = float(safe_get(r, 1, "0") or 0)
        max_depth = float(safe_get(r, 2, "0") or 0)
        x, y = coords.get(nid, ("", ""))
        node_rows.append({
            "ObjectTable": "hw_node", "node_id": nid, "node_type": "StorageUnit",
            "asset_id": "Storage", "system_type": "combined",
            "x": x, "y": y,
            "chamber_floor": elev, "chamber_roof": elev + max_depth,
            "ground_level": elev + max_depth, "flood_level": elev + max_depth,
            "flood_type": "Stored",
        })
    write_csv(out_dir / "BellingeSWMM_node.csv", NODE_COLS, node_rows)

    # ------ CONDUITS -------------------------------------------------
    cond_rows: list[dict] = []
    # CONDUITS: Name FromNode ToNode Length Roughness InOffset OutOffset InitFlow MaxFlow
    for r in sec.get("CONDUITS", []):
        name = r[0]
        us_node, ds_node = r[1], r[2]
        length = float(r[3])
        rough = float(r[4])
        in_off = float(r[5])
        out_off = float(r[6])

        # Look up xsection to fill geometry
        xs = xsect.get(name, [])
        shape_raw = (xs[0] if xs else "").upper()
        # Map SWMM shapes to InfoWorks-ish names
        shape_map = {"CIRCULAR": "CIRC", "RECT_CLOSED": "RECT",
                     "RECT_OPEN": "RECT", "EGG": "EGG",
                     "FORCE_MAIN": "CIRC", "FILLED_CIRCULAR": "CIRC"}
        shape = shape_map.get(shape_raw, shape_raw)
        geom1 = safe_float(xs[1]) if len(xs) > 1 else 0.0  # diameter / height
        geom2 = safe_float(xs[2]) if len(xs) > 2 else 0.0  # width or curve_name (CUSTOM)

        if shape_raw == "CIRCULAR" or shape_raw.endswith("CIRCULAR"):
            width = geom1
            height = geom1
        else:
            height = geom1
            width = geom2

        # SWMM offsets are depths above invert (LINK_OFFSETS=DEPTH)
        # us_invert is reconstructed from the upstream node invert + offset
        # We don't have node-invert lookup here; downstream code can recompute.
        gradient = "" if length == 0 else "{:.6f}".format(0.0)  # placeholder

        cond_rows.append({
            "ObjectTable": "hw_conduit",
            "us_node_id": us_node,
            "link_suffix": "1",
            "ds_node_id": ds_node,
            "link_type": "Cond",
            "asset_id": name,
            "system_type": "combined",
            "shape": shape,
            "conduit_width": width,
            "conduit_height": height,
            "conduit_length": length,
            "us_invert": in_off,   # offset from upstream-node invert (m)
            "ds_invert": out_off,  # offset from downstream-node invert (m)
            "bottom_roughness_Manning": rough,
            "top_roughness_Manning": rough,
            "roughness_type": "Manning",
            "number_of_barrels": 1,
            "gradient": gradient,
        })
    write_csv(out_dir / "BellingeSWMM_conduit.csv", CONDUIT_COLS, cond_rows)

    # ------ PUMPS ----------------------------------------------------
    pump_rows: list[dict] = []
    # PUMPS: Name FromNode ToNode PumpCurve Status Startup Shutoff
    for r in sec.get("PUMPS", []):
        name = r[0]
        us_node, ds_node = r[1], r[2]
        curve_id = safe_get(r, 3)
        startup = float(safe_get(r, 5, "0") or 0)
        shutoff = float(safe_get(r, 6, "0") or 0)
        pump_rows.append({
            "ObjectTable": "hw_pump",
            "us_node_id": us_node,
            "link_suffix": "1",
            "ds_node_id": ds_node,
            "link_type": "PUMP",
            "system_type": "combined",
            "asset_id": name,
            "switch_on_level": startup,
            "switch_off_level": shutoff,
            "head_discharge_id": curve_id,
        })
    write_csv(out_dir / "BellingeSWMM_pump.csv", PUMP_COLS, pump_rows)

    # ------ WEIRS ----------------------------------------------------
    weir_rows: list[dict] = []
    # WEIRS: Name FromNode ToNode Type CrestHt Qcoeff Gated EndCon EndCoeff Surcharge ...
    for r in sec.get("WEIRS", []):
        name = r[0]
        us_node, ds_node = r[1], r[2]
        wtype = safe_get(r, 3)
        crest = float(safe_get(r, 4, "0") or 0)
        qcoeff = float(safe_get(r, 5, "0") or 0)
        # Width/height from XSECTIONS for weirs
        xs = xsect.get(name, [])
        height = safe_float(xs[1]) if len(xs) > 1 else 0.0
        width = safe_float(xs[2]) if len(xs) > 2 else 0.0
        weir_rows.append({
            "ObjectTable": "hw_weir",
            "us_node_id": us_node,
            "link_suffix": "1",
            "ds_node_id": ds_node,
            "link_type": "WEIR",
            "system_type": "combined",
            "asset_id": wtype,
            "crest": crest,
            "width": width,
            "height": height,
            "discharge_coeff": qcoeff,
            "orientation": "Forward",
        })
    write_csv(out_dir / "BellingeSWMM_weir.csv", WEIR_COLS, weir_rows)

    # ------ ORIFICES -------------------------------------------------
    orif_rows: list[dict] = []
    # ORIFICES: Name FromNode ToNode Type Offset Qcoeff Gated CloseTime
    for r in sec.get("ORIFICES", []):
        name = r[0]
        us_node, ds_node = r[1], r[2]
        offset = float(safe_get(r, 4, "0") or 0)
        qcoeff = float(safe_get(r, 5, "0") or 0)
        xs = xsect.get(name, [])
        diameter = safe_float(xs[1]) if len(xs) > 1 else 0.0
        orif_rows.append({
            "ObjectTable": "hw_orifice",
            "us_node_id": us_node,
            "link_suffix": "1",
            "ds_node_id": ds_node,
            "link_type": "ORIFIC",
            "system_type": "combined",
            "asset_id": name,
            "invert": offset,
            "diameter": diameter,
            "discharge_coeff": qcoeff,
            "opening_type": "Rond",
        })
    write_csv(out_dir / "BellingeSWMM_orifice.csv", ORIFICE_COLS, orif_rows)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inp", default="data/raw/bellinge/7_SWMM/BellingeSWMM_v021_nopervious.inp")
    p.add_argument("--out", default="data/interim/bellinge/topology")
    args = p.parse_args()
    main(Path(args.inp), Path(args.out))
