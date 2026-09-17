"""Section 5.1: simulated vs measured water level at Bellinge manhole G72F040.

G72F040 is the only cleaned Bellinge sensor (1-minute, Jan-Mar 2020) that sits
on a manhole rather than a storage unit, and it is not in the 1% sensor set.

Both series are water level in m above datum; the sensor level is its depth
plus the 19.28 m zero point (Table 1 of 2_Sensordata_v2.pdf).

    python -m analysis.real_sensor.sim_vs_measured
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SENSOR = "G72F040"
ZERO_POINT_M = 19.28          # Table 1, 2_Sensordata_v2.pdf
CLEANED = REPO / "data/raw/bellinge/2_cleaned_data/G72F040_Danovap1_proc_v6.csv"
EVENTS = REPO / "data/interim/bellinge/event_list.csv"
SIMDIR = REPO / "data/interim/bellinge/simulations"


def load_measured() -> pd.Series:
    d = pd.read_csv(CLEANED, usecols=["time", "depth_s", "level"], parse_dates=["time"])
    lvl = pd.to_numeric(d["level"], errors="coerce")
    # fall back to depth + 0-point where the level column is absent
    dep = pd.to_numeric(d["depth_s"], errors="coerce")
    lvl = lvl.fillna(dep + ZERO_POINT_M)
    s = pd.Series(lvl.values, index=d["time"]).dropna()
    return s[~s.index.duplicated()]


def load_simulated(tag: str) -> pd.Series | None:
    f = glob.glob(str(SIMDIR / f"*{tag}_depnod.csv"))
    if not f:
        return None
    df = pd.read_csv(f[0], skiprows=[1], low_memory=False)   # row 1 is the unit row
    if SENSOR not in df.columns:
        return None
    t = pd.to_datetime(df["Time"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
    v = pd.to_numeric(df[SENSOR], errors="coerce")
    # Some event files carry 0.0 as a missing-value sentinel -- 40% of e1199, for
    # instance. A level of 0 m AD is impossible at a manhole whose invert sits at
    # 19.28 m, so anything below the invert is dropped as missing rather than
    # averaged in; two events would otherwise contribute 500-800 cm "errors" and
    # dominate every pooled statistic.
    v = v.where(v > ZERO_POINT_M - 0.5)
    s = pd.Series(v.values, index=t).dropna()
    return s[~s.index.duplicated()]


def metrics(sim: np.ndarray, obs: np.ndarray) -> dict:
    e = sim - obs
    var = ((obs - obs.mean()) ** 2).sum()
    return dict(n=int(len(e)),
                mae_cm=float(np.abs(e).mean() * 100),
                rmse_cm=float(np.sqrt((e ** 2).mean()) * 100),
                bias_cm=float(e.mean() * 100),
                nse=float(1 - (e ** 2).sum() / var) if var > 0 else float("nan"),
                r=float(np.corrcoef(sim, obs)[0, 1]) if len(e) > 2 else float("nan"),
                obs_range_cm=float((obs.max() - obs.min()) * 100))


def main():
    obs = load_measured()
    print(f"measured {SENSOR}: {len(obs)} readings, "
          f"{obs.index.min()} -> {obs.index.max()}")
    print(f"  level {obs.min():.3f}-{obs.max():.3f} m AD "
          f"(depth {obs.min()-ZERO_POINT_M:.3f}-{obs.max()-ZERO_POINT_M:.3f} m)")

    ev = pd.read_csv(EVENTS, parse_dates=["start", "end"])
    win = ev[(ev.start >= obs.index.min().normalize()) & (ev.end <= obs.index.max())]
    print(f"\nsimulated events inside the measured window: {len(win)}")

    rows, pooled_s, pooled_o = [], [], []
    for _, e in win.iterrows():
        sim = load_simulated(e.tag)
        if sim is None or sim.empty:
            continue
        # compare on the simulation's own stamps, nearest measurement within 1 min
        j = obs.reindex(sim.index, method="nearest", tolerance=pd.Timedelta("1min")).dropna()
        if len(j) < 30:
            continue
        s = sim.reindex(j.index).values.astype(float)
        o = j.values.astype(float)
        m = metrics(s, o); m["tag"] = e.tag; m["rain_mm"] = float(e.total_mm)
        m["coverage"] = float(len(j) / max(len(sim), 1))
        rows.append(m); pooled_s.append(s); pooled_o.append(o)

    if not rows:
        raise SystemExit("no overlapping events could be aligned")

    print(f"{'event':>12} {'rain':>7} {'n':>6} {'MAE':>8} {'RMSE':>8} "
          f"{'bias':>8} {'NSE':>7} {'r':>6}")
    for m in rows:
        print(f"{m['tag']:>12} {m['rain_mm']:7.1f} {m['n']:6d} {m['mae_cm']:8.2f} "
              f"{m['rmse_cm']:8.2f} {m['bias_cm']:+8.2f} {m['nse']:7.3f} {m['r']:6.3f}")

    S, O = np.concatenate(pooled_s), np.concatenate(pooled_o)
    P = metrics(S, O)
    print(f"\nPOOLED over {len(rows)} events, {P['n']} aligned timesteps:")
    print(f"  simulator vs measurement   MAE {P['mae_cm']:.2f} cm   RMSE {P['rmse_cm']:.2f} cm"
          f"   bias {P['bias_cm']:+.2f} cm   NSE {P['nse']:.3f}   r {P['r']:.3f}")
    print(f"  measured level range       {P['obs_range_cm']:.1f} cm")
    print("\nRead this as the accuracy ceiling for anything trained on these targets:")
    print("a model reproducing the simulation perfectly would still differ from the")
    print("real manhole by roughly this much.")

    out = REPO / "results/real_sensor_sim_vs_measured.json"
    out.write_text(json.dumps({"sensor": SENSOR, "zero_point_m": ZERO_POINT_M,
                               "pooled": P, "events": rows}, indent=2))
    print(f"\nWritten {out}")


if __name__ == "__main__":
    main()
