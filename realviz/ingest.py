"""Ingest a real Pulsoid export into the 10 s grid the detector expects.

Pulsoid's schema is not the one generate_day.py emits. It is one row per beat
report, every ~5 s but irregular, every field quoted:

    "beats_per_minute","unix_millis","date_time_utc","date_time_America/New_York"

Three things have to happen before any derivative is worth taking.

1.  HARMONIC ARTIFACTS.  An optical sensor that loses the pulse will sometimes
    lock onto the second harmonic and report exactly double. This recording has
    four such runs -- 70 -> 136 inside a single 4 s sample, held for a minute,
    then back to 70. Left in, each one is a +990 bpm/min edge: the largest
    "stress onset" in the file would be a sensor glitch. They are caught on two
    independent signatures, both of which must fire (see mask_harmonics).

2.  IRREGULAR SPACING.  Samples land 3-7 s apart, sometimes twice in the same
    second. Savitzky-Golay assumes a constant delta, so the series is binned
    onto a uniform 10 s grid first.

3.  REAL DROPOUTS.  36 gaps exceed 30 s, the longest ~5 min. Those bins stay
    empty. They are not interpolated and the detector abstains there.
"""
import pathlib
import numpy as np
import pandas as pd

GRID_S = 10           # target grid, matches the synthetic pipeline
MAX_SLEW = 120.0      # bpm/min. a resting heart cannot climb faster than this
HARM_LO, HARM_HI = 1.72, 2.30   # ratio-to-baseline band that means "doubled"
BASE_WIN = 41         # samples (~3.5 min) for the local baseline median


def read_pulsoid(path):
    df = pd.read_csv(path)
    need = {"beats_per_minute", "unix_millis", "date_time_utc"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{path}: not a Pulsoid export, missing {sorted(missing)}")

    local_col = next((c for c in df.columns if c.startswith("date_time_")
                      and c != "date_time_utc"), None)
    df["hr"] = df["beats_per_minute"].astype(float)
    df["t_utc"] = pd.to_datetime(df["date_time_utc"], utc=True)
    # display on the wearer's own clock -- the export already carries it, so we
    # never have to guess a zone
    # Pulsoid drops ":00" seconds, so the local column is mixed-width ISO
    df["t"] = (pd.to_datetime(df[local_col], format="ISO8601") if local_col
               else df["t_utc"].dt.tz_localize(None))
    df["tz"] = local_col.replace("date_time_", "") if local_col else "UTC"
    return df.sort_values("t").drop_duplicates("t_utc").reset_index(drop=True)


def mask_harmonics(df):
    """Flag double-counted runs. Both signatures must agree.

    ratio  -- HR sits at 1.7-2.3x a local median that excludes the run itself
    slew   -- the run is entered or left faster than a heart can actually move

    Requiring both keeps genuine tachycardia: a real climb to 2x baseline takes
    tens of seconds, so it never trips the slew test, and a fast but small step
    never reaches the ratio band.
    """
    hr = df["hr"]
    base = hr.rolling(BASE_WIN, center=True, min_periods=5).median()
    ratio = hr / base
    doubled = ratio.between(HARM_LO, HARM_HI)

    dt = df["t_utc"].diff().dt.total_seconds().replace(0, np.nan)
    slew = (hr.diff() / dt * 60.0).abs()

    # a run is contiguous doubled samples; keep it only if some edge is impossible
    grp = (doubled != doubled.shift()).cumsum()
    bad = pd.Series(False, index=df.index)
    for _, run in df.groupby(grp).groups.items():
        idx = pd.Index(run)
        if not doubled.loc[idx].iloc[0]:
            continue
        edges = [idx[0]]
        if idx[-1] + 1 in slew.index:
            edges.append(idx[-1] + 1)
        if slew.loc[edges].max() > MAX_SLEW:
            bad.loc[idx] = True

    df["artifact"] = bad.to_numpy()
    return df


def to_grid(df):
    """Bin onto a uniform GRID_S lattice. Empty bins stay NaN."""
    clean = df.loc[~df["artifact"], ["t", "hr"]].set_index("t")
    t0 = df["t"].iloc[0].floor(f"{GRID_S}s")
    t1 = df["t"].iloc[-1].ceil(f"{GRID_S}s")
    grid = pd.date_range(t0, t1, freq=f"{GRID_S}s")

    # median inside the bin: robust to the occasional duplicate-timestamp pair
    binned = clean["hr"].resample(f"{GRID_S}s", origin=t0).median()
    out = pd.DataFrame({"timestamp": grid})
    out["hr_bpm"] = binned.reindex(grid).to_numpy()

    # keep the masked samples for the viewer so the glitch is visible, not hidden
    art = df.loc[df["artifact"], ["t", "hr"]].set_index("t")["hr"]
    out["hr_artifact"] = art.resample(f"{GRID_S}s", origin=t0).median() \
                            .reindex(grid).to_numpy()
    return out


def main(src=None, dst="data/real/hr_10s.csv"):
    src = src or sorted(pathlib.Path(".").glob("pulsoid-report-*.csv"))[-1]
    raw = mask_harmonics(read_pulsoid(src))
    out = to_grid(raw)

    pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst, index=False, date_format="%Y-%m-%d %H:%M:%S")

    span = (raw["t"].iloc[-1] - raw["t"].iloc[0]).total_seconds()
    filled = out["hr_bpm"].notna().sum()
    print(f"source        {pathlib.Path(src).name}")
    print(f"timezone      {raw['tz'].iloc[0]}")
    print(f"raw samples   {len(raw):,}  over {span/3600:.2f} h "
          f"(median spacing {raw['t_utc'].diff().dt.total_seconds().median():.0f} s)")
    print(f"harmonics     {int(raw['artifact'].sum())} samples masked in "
          f"{int((raw['artifact'] & ~raw['artifact'].shift(fill_value=False)).sum())} runs")
    print(f"grid          {len(out):,} bins @ {GRID_S}s  "
          f"({filled:,} filled, {len(out)-filled:,} dropout = "
          f"{(len(out)-filled)/len(out)*100:.1f}%)")
    print(f"HR            {out.hr_bpm.min():.0f}-{out.hr_bpm.max():.0f} bpm, "
          f"median {out.hr_bpm.median():.0f}")
    print(f"-> {dst}")
    return out


if __name__ == "__main__":
    main()
