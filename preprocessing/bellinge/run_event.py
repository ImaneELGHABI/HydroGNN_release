#!/usr/bin/env python3
"""Run the Bellinge SWMM model for one event window.

Writes a depnod CSV (Time, Seconds, <node_id>...) and optionally a link flow
CSV, read from the binary .out file via swmm-toolkit.

    python preprocessing/bellinge/run_event.py --start 2012-06-29 --end 2012-06-30 --tag bui#01

Without --start/--end the dates in the .inp are used.
"""
from __future__ import annotations
import argparse, csv, re, shutil, time
from datetime import datetime, timedelta
from pathlib import Path

from pyswmm import Simulation
from swmm.toolkit import output, shared_enum

SRC_INP = Path("data/raw/bellinge/7_SWMM/BellingeSWMM_v021_nopervious.inp")
RAIN_DAT = SRC_INP.parent / "rg_bellinge_Jun2010_Aug2021.dat"


def patch_inp_dates(src: Path, dst: Path, start: datetime | None, end: datetime | None) -> None:
    """Copy SRC to DST replacing START_DATE/END_DATE/REPORT_START_DATE if given."""
    text = src.read_text(encoding="latin1")
    if start is not None:
        s = start.strftime("%m/%d/%Y")
        text = re.sub(r"^(START_DATE\s+)\S+", rf"\g<1>{s}", text, flags=re.M)
        text = re.sub(r"^(REPORT_START_DATE\s+)\S+", rf"\g<1>{s}", text, flags=re.M)
        text = re.sub(r"^(START_TIME\s+)\S+", r"\g<1>00:00:00", text, flags=re.M)
        text = re.sub(r"^(REPORT_START_TIME\s+)\S+", r"\g<1>00:00:00", text, flags=re.M)
    if end is not None:
        e = end.strftime("%m/%d/%Y")
        text = re.sub(r"^(END_DATE\s+)\S+", rf"\g<1>{e}", text, flags=re.M)
        text = re.sub(r"^(END_TIME\s+)\S+", r"\g<1>23:59:00", text, flags=re.M)
    dst.write_text(text, encoding="latin1")


def run_swmm(inp_path: Path) -> tuple[Path, Path]:
    """Run SWMM end-to-end and return (out_path, rpt_path)."""
    print(f"  running SWMM ...", flush=True)
    t0 = time.time()
    with Simulation(str(inp_path)) as sim:
        for _ in sim:
            pass
    dt = time.time() - t0
    out = inp_path.with_suffix(".out")
    rpt = inp_path.with_suffix(".rpt")
    if not out.exists():
        raise RuntimeError(f"SWMM produced no .out file at {out}")
    print(f"  done in {dt:.1f}s  →  {out.name} ({out.stat().st_size/1e6:.1f} MB)", flush=True)
    return out, rpt


# Heusden depnod shape:
#   Time, Seconds, <node1>, <node2>, ...
#   [Hr:Min:s], [s], depnod [m AD], depnod [m AD], ...
#   00/00/0000 00:00:00, 0, 1.12044, 1.04088, ...
def write_depnod(out_path: Path, dst_csv: Path, sim_start: datetime,
                 node_ids: list[str], invert_levels: dict[str, float]) -> int:
    """Read node depths from .out and write a Heusden-shaped depnod CSV.

    SWMM's NODE_DEPTH attribute is depth above invert. Heusden's depnod is
    `m AD` (water level above datum). We add the invert level to obtain a
    comparable absolute level.
    """
    handle = output.init()
    try:
        output.open(handle, str(out_path))
        n_periods = output.get_times(handle, shared_enum.Time.NUM_PERIODS)
        report_step = output.get_times(handle, shared_enum.Time.REPORT_STEP)  # seconds
        n_nodes = output.get_proj_size(handle)[1]
        swmm_nodes = [output.get_elem_name(handle, shared_enum.ElementType.NODE, i)
                      for i in range(n_nodes)]
        idx = {n: i for i, n in enumerate(swmm_nodes)}
        # Filter requested ids to those SWMM actually has
        keep = [n for n in node_ids if n in idx]
        missing = [n for n in node_ids if n not in idx]
        if missing:
            print(f"  WARN: {len(missing)} requested nodes not in .out (e.g. {missing[:5]})")

        # Buffer the whole timeseries: (n_periods, len(keep))
        # output.get_node_series returns one node's full timeseries in one call
        depths_per_node = {}
        for n in keep:
            depths_per_node[n] = output.get_node_series(
                handle, idx[n], shared_enum.NodeAttribute.HYDRAULIC_HEAD, 0, n_periods - 1
            )
            # HYDRAULIC_HEAD already includes invert (it's the absolute water surface
            # elevation), which matches Heusden depnod [m AD]. No need to add invert.

        # Write CSV
        dst_csv.parent.mkdir(parents=True, exist_ok=True)
        with dst_csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Time", "Seconds", *keep])
            w.writerow(["[Hr:Min:s]", "[s]", *[" depnod [m AD]"] * len(keep)])
            for t in range(n_periods):
                # SWMM output ts is at sim_start + (t+1) * report_step
                cur = sim_start + timedelta(seconds=(t + 1) * report_step)
                row = [cur.strftime("%d/%m/%Y %H:%M:%S"),
                       (t + 1) * report_step]
                row.extend(f"{depths_per_node[n][t]:.5f}" for n in keep)
                w.writerow(row)
        return n_periods
    finally:
        output.close(handle)


def get_node_ids_from_topology(topo_csv: Path) -> tuple[list[str], dict[str, float]]:
    """Read node_id and chamber_floor from the topology node CSV."""
    ids = []
    inv = {}
    with topo_csv.open() as f:
        rd = csv.DictReader(f)
        for r in rd:
            ids.append(r["node_id"])
            try:
                inv[r["node_id"]] = float(r["chamber_floor"])
            except (KeyError, ValueError):
                inv[r["node_id"]] = 0.0
    return ids, inv


def main(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-event isolated work dir to avoid race conditions across parallel tasks
    work = out_dir / f"_swmm_work_{args.tag}"
    work.mkdir(exist_ok=True)

    # Stage .inp + symlink rain file
    staged_inp = work / SRC_INP.name
    start = datetime.fromisoformat(args.start) if args.start else None
    end = datetime.fromisoformat(args.end) if args.end else None
    patch_inp_dates(SRC_INP, staged_inp, start, end)

    rain_link = work / RAIN_DAT.name
    if rain_link.exists() or rain_link.is_symlink():
        rain_link.unlink()
    rain_link.symlink_to(RAIN_DAT)

    print(f"Event tag: {args.tag}")
    print(f"Window:    {start} → {end}")

    existing_out = staged_inp.with_suffix(".out")
    if args.skip_run and existing_out.exists():
        print(f"  reusing existing {existing_out.name} ({existing_out.stat().st_size/1e6:.1f} MB)")
        out = existing_out
    else:
        out, _ = run_swmm(staged_inp)

    # Choose node order from topology CSV (so depnod columns line up with topology)
    topo = Path(args.topology_dir) / "BellingeSWMM_node.csv"
    node_ids, inverts = get_node_ids_from_topology(topo)
    print(f"  topology has {len(node_ids)} nodes")

    # Determine effective sim_start (the patched start, or the original from .inp)
    if start is None:
        m = re.search(r"^START_DATE\s+(\S+)", staged_inp.read_text(encoding="latin1"), re.M)
        start = datetime.strptime(m.group(1), "%m/%d/%Y")

    dst = out_dir / f"Node_BellingeSWMM_{args.tag}_depnod.csv"
    n = write_depnod(out, dst, start, node_ids, inverts)
    print(f"  wrote {dst.name}  ({n} timesteps × {len(node_ids)} nodes)")

    # Cleanup
    if not args.keep_intermediate:
        for ext in (".inp", ".out", ".rpt"):
            p = staged_inp.with_suffix(ext)
            if p.exists():
                p.unlink(missing_ok=True)
        if rain_link.exists() or rain_link.is_symlink():
            rain_link.unlink()
        try:
            work.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", default=None, help="YYYY-MM-DD; default = .inp value")
    p.add_argument("--end", default=None, help="YYYY-MM-DD; default = .inp value")
    p.add_argument("--tag", default="bui01", help="event tag in output filename")
    p.add_argument("--out_dir", default="data/interim/bellinge/simulations")
    p.add_argument("--topology_dir", default="data/interim/bellinge/topology")
    p.add_argument("--keep_intermediate", action="store_true")
    p.add_argument("--skip_run", action="store_true",
                   help="reuse the .out already in <out_dir>/_swmm_work/")
    main(p.parse_args())
