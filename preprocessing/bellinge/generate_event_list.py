#!/usr/bin/env python3
"""Extract storm events from the Bellinge rain record into event_list.csv.

Splits by year: train 2009-2018, val 2019-2020, test 2021 onward.

Each window is padded:
  start = rain start - WARMUP_H hours
  end   = rain end   + TAIL_H hours
"""
from pathlib import Path
import pandas as pd

REPO      = Path(__file__).resolve().parents[2]
RAIN_DAT  = REPO / "data/raw/bellinge/7_SWMM/rg_bellinge_Jun2010_Aug2021.dat"
OUT_CSV   = REPO / "data/interim/bellinge/event_list.csv"

MIN_MM        = 2.0   # minimum total rainfall [mm]
MIN_STEPS     = 30    # minimum timesteps in rain record
GAP_H         = 6     # inter-event gap threshold [hours]
WARMUP_H      = 6     # hours of dry-run before rain starts
TAIL_H        = 6     # hours of drainage after rain ends


def main():
    print(f"Reading {RAIN_DAT} …")
    df = pd.read_csv(RAIN_DAT, sep=r"\s+", header=None,
                     names=["gauge","year","month","day","hour","minute","mm"])
    df["dt"] = pd.to_datetime(dict(year=df.year, month=df.month, day=df.day,
                                   hour=df.hour, minute=df.minute))
    df = df.sort_values("dt").reset_index(drop=True)
    df["gap_h"] = df["dt"].diff().dt.total_seconds().fillna(0) / 3600

    # Identify event boundaries
    event_ids = (df["gap_h"] > GAP_H).cumsum()
    events = df.groupby(event_ids).agg(
        rain_start=("dt", "first"),
        rain_end  =("dt", "last"),
        total_mm  =("mm", "sum"),
        n_steps   =("mm", "count"),
    ).reset_index(drop=True)

    # Filter significant events
    sig = events[(events.total_mm >= MIN_MM) & (events.n_steps >= MIN_STEPS)].copy()
    sig = sig.sort_values("rain_start").reset_index(drop=True)

    # Pad with warmup/tail
    sig["start"] = sig["rain_start"] - pd.Timedelta(hours=WARMUP_H)
    sig["end"]   = sig["rain_end"]   + pd.Timedelta(hours=TAIL_H)

    # Year-based split
    def assign_split(row):
        y = row["rain_start"].year
        if y <= 2018: return "train"
        if y <= 2020: return "val"
        return "test"

    sig["split"] = sig.apply(assign_split, axis=1)

    # Tag: e{index:04d}_{split}
    sig["tag"] = [f"e{i:04d}_{row.split}" for i, row in sig.iterrows()]

    # Output
    out = sig[["tag", "split", "start", "end", "total_mm", "n_steps"]].copy()
    out["start"] = out["start"].dt.strftime("%Y-%m-%d")
    out["end"]   = out["end"].dt.strftime("%Y-%m-%d")
    out.to_csv(OUT_CSV, index=False)

    print(f"\nEvent summary:")
    for split in ["train", "val", "test"]:
        s = out[out.split == split]
        print(f"  {split:5s}: {len(s):4d} events  "
              f"rainfall {sig.loc[out.split==split,'total_mm'].mean():.1f} mm avg")
    print(f"  TOTAL: {len(out)} events")
    print(f"\n✓ Written → {OUT_CSV}")


if __name__ == "__main__":
    main()
