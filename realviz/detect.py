"""Stress scoring for the real Pulsoid recording, and for the synthetic walk.

Same rule as stress_detect.py:

    if  dS/dt < alpha    (movement cannot explain it)
    and dHR/dt > beta    (heart rate is climbing anyway)
    -> stress_score  proportional to  dHR/dt

with two changes forced by real data.

BETA IS CALIBRATED, NOT INHERITED.  beta on the synthetic day was set from that
recording's noise floor. This is a different sensor at a different sampling
rate with 18.5% dropout, so its floor is its own number. It is measured here
with a median-absolute-deviation sigma -- MAD ignores the genuine excursions
instead of being inflated by them, which a plain standard deviation would be --
and beta is placed at BETA_SIGMA times that floor.

THE REAL TRACK HAS NO STEPS.  Pulsoid streams heart rate only. There is no
honest way to recover cadence from it, so none is invented: the stillness gate
on the real track is *assumed open* and the page says so on every panel. That
makes the real score an upper bound -- every rise is charged as stress, whether
or not the wearer was moving. The synthetic walk is what shows the size of
that assumption: run the same detector with a real cadence channel and the
walking rises, which are the largest in the file, drop out entirely.
"""
import json
import pathlib
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ALPHA = 10.0        # steps/min. below this you are, for our purposes, still
BETA_SIGMA = 3.3    # where beta sits on the measured noise floor
SCALE = 25.0        # bpm/min mapped to a score of 100
DEBOUNCE = 2        # consecutive windows to fire
MAX_GAP = 3         # bridge dropouts up to 30 s; longer and we abstain
MERGE_GAP = 6       # episodes closer than a minute are one episode
GRID_S = 10
DT_MIN = GRID_S / 60.0
SAVGOL_W = 7        # 7 samples = 60 s


def prepare(df, hr_col="hr_bpm"):
    """Smooth, bridge short gaps, take dHR/dt over a 60 s window."""
    hr = df[hr_col].astype(float)

    # a dropout longer than MAX_GAP is not interpolated -- the detector
    # abstains there rather than inventing a slope across missing data
    isna = hr.isna()
    grp = (isna != isna.shift()).cumsum()
    runlen = isna.groupby(grp).transform("size")
    df["valid"] = ~(isna & (runlen > MAX_GAP))

    f = hr.interpolate(limit=MAX_GAP, limit_direction="both")
    f = f.rolling(3, center=True, min_periods=1).median()
    # forward then backward EMA: zero net phase lag, so onsets are not
    # reported late
    f = f.ewm(span=5, adjust=False).mean()
    f = f[::-1].ewm(span=5, adjust=False).mean()[::-1]
    df["hr_smooth"] = f

    filled = f.bfill().ffill().to_numpy()
    df["dHR_dt"] = savgol_filter(filled, SAVGOL_W, 2, deriv=1,
                                 delta=float(GRID_S)) * 60.0
    df.loc[~df["valid"], "dHR_dt"] = np.nan
    return df


def calibrate_beta(df):
    """Noise floor of dHR/dt, from the recording itself.

    MAD -> sigma via the 1.4826 consistency constant for a normal. Robust to
    the real climbs in the trace, which is the whole point: we want the floor,
    not the spread.
    """
    d = df.loc[df["valid"], "dHR_dt"].dropna().to_numpy()
    med = float(np.median(d))
    sigma = float(np.median(np.abs(d - med)) * 1.4826)
    return {"median": med, "sigma": sigma,
            "beta": round(med + BETA_SIGMA * sigma, 2)}


def cadence(df):
    """dS/dt in steps per minute, centred on a 60 s window."""
    df["cadence_spm"] = df["steps"].rolling(6, center=True,
                                            min_periods=1).mean() * 6.0
    return df


def detect(df, beta, alpha=ALPHA, debounce=DEBOUNCE, gated=True):
    """gated=False is the real track: no step channel, gate assumed open."""
    still = (df["cadence_spm"] < alpha) if gated else pd.Series(True, index=df.index)
    rising = df["dHR_dt"] > beta
    raw = (still & rising & df["valid"]).fillna(False)

    fired = raw.copy()
    for k in range(1, debounce):
        fired &= raw.shift(k, fill_value=False)
    for k in range(1, debounce):          # credit the whole run, not its tail
        fired |= fired.shift(-k, fill_value=False)

    df["gate_open"] = still & df["valid"]
    df["is_stress"] = fired
    df["stress_score"] = np.where(
        fired, np.clip(df["dHR_dt"] / SCALE * 100.0, 0, 100), 0.0)
    return df


def episodes(df):
    idx = np.flatnonzero(df["is_stress"].to_numpy())
    if idx.size == 0:
        return pd.DataFrame(columns=["start", "end", "dur_min", "peak_score",
                                     "peak_dHR_dt", "hr_rise", "area"])
    splits = np.flatnonzero(np.diff(idx) > MERGE_GAP) + 1
    out = []
    for run in np.split(idx, splits):
        a, b = int(run[0]), int(run[-1])
        seg = df.iloc[a:b + 1]
        hrs = seg["hr_smooth"].dropna()
        out.append({
            "start": df["timestamp"].iloc[a],
            "end": df["timestamp"].iloc[b],
            "dur_min": round((b - a + 1) * DT_MIN, 2),
            "peak_score": round(float(seg["stress_score"].max()), 1),
            "peak_dHR_dt": round(float(seg["dHR_dt"].max()), 1),
            "hr_rise": round(float(hrs.iloc[-1] - hrs.iloc[0]), 1) if len(hrs) > 1 else 0.0,
            "area": round(float(seg["stress_score"].sum()) * DT_MIN, 1),
        })
    return pd.DataFrame(out)


def score_vs_truth(df, gt):
    """Per-sample precision/recall against the synthetic walk's answer key.

    stress_sustained is elevated-but-flat. A derivative detector structurally
    cannot see it, so it is excluded from the numbers rather than counted as a
    miss, and reported on its own.
    """
    kind = pd.Series(["baseline"] * len(df), index=df.index)
    for _, r in gt.iterrows():
        m = (df["timestamp"] >= r["start"]) & (df["timestamp"] < r["end"])
        kind[m] = r["kind"]
    df["gt_kind"] = kind
    pred = df["is_stress"].to_numpy()
    scored = (kind != "stress_sustained").to_numpy()
    truth = (kind == "stress_onset").to_numpy()
    tp = int((pred & truth & scored).sum())
    fp = int((pred & ~truth & scored).sum())
    fn = int((~pred & truth & scored).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else None,
            "fp_while_walking": int((pred & (kind == "activity").to_numpy()).sum()),
            "fp_negative_control": int((pred & (kind == "negative_control").to_numpy()).sum()),
            "onsets": int(gt["kind"].eq("stress_onset").sum()),
            "onsets_hit": int(sum(
                bool(df.loc[(df["timestamp"] >= r["start"]) &
                            (df["timestamp"] < r["end"]), "is_stress"].any())
                for _, r in gt[gt["kind"] == "stress_onset"].iterrows()))}


def hr_context(df):
    """Resting HR and HR reserve -- the denominators a score needs to mean
    anything across people. Resting HR is the 5th percentile of the smoothed
    trace, which is the usual wearable convention."""
    hr = df["hr_smooth"].dropna()
    rest = float(np.percentile(hr, 5))
    return {"resting_bpm": round(rest, 1),
            "median_bpm": round(float(hr.median()), 1),
            "peak_bpm": round(float(hr.max()), 1),
            "reserve_used_pct": round((float(hr.max()) - rest) /
                                      max(220 - 22 - rest, 1) * 100, 1)}


def run():
    real = pd.read_csv("data/real/hr_10s.csv", parse_dates=["timestamp"])
    walk = pd.read_csv("data/real/walk_10s.csv", parse_dates=["timestamp"])
    gt = pd.read_csv("data/real/walk_events.csv", parse_dates=["start", "end"])

    real = prepare(real)
    cal = calibrate_beta(real)
    beta = cal["beta"]

    real["cadence_spm"] = np.nan
    real = detect(real, beta, gated=False)

    walk = cadence(prepare(walk))
    walk = detect(walk, beta, gated=True)

    er, ew = episodes(real), episodes(walk)
    stats = score_vs_truth(walk, gt)

    # the size of the ungated assumption: rerun the walk with its gate removed
    ungated = detect(walk.copy(), beta, gated=False)

    summary = {
        "beta": beta, "beta_sigma": BETA_SIGMA, "alpha": ALPHA,
        "scale": SCALE, "debounce": DEBOUNCE, "noise_floor": cal,
        "real": {
            "n": len(real), "hours": round(len(real) * GRID_S / 3600, 2),
            "dropout_pct": round(float(real["hr_bpm"].isna().mean()) * 100, 1),
            "abstain_pct": round(float((~real["valid"]).mean()) * 100, 1),
            "episodes": len(er),
            "load": round(float(real["stress_score"].sum()) * DT_MIN, 1),
            "minutes_flagged": round(float(real["is_stress"].sum()) * DT_MIN, 1),
            **hr_context(real)},
        "walk": {
            "n": len(walk), "steps": int(walk["steps"].sum()),
            "walk_minutes": round(float((walk["cadence_spm"] > 30).sum()) * DT_MIN, 1),
            "episodes": len(ew),
            "load": round(float(walk["stress_score"].sum()) * DT_MIN, 1),
            "load_ungated": round(float(ungated["stress_score"].sum()) * DT_MIN, 1),
            "episodes_ungated": len(episodes(ungated)),
            **stats, **hr_context(walk)},
    }

    out = pathlib.Path("data/real")
    real.to_csv(out / "timeline_real.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    walk.to_csv(out / "timeline_walk.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    er.to_csv(out / "episodes_real.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    ew.to_csv(out / "episodes_walk.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return real, walk, gt, er, ew, summary


def main():
    real, walk, gt, er, ew, s = run()
    print(f"NOISE FLOOR (real)   dHR/dt = {s['noise_floor']['median']:+.2f} "
          f"+/- {s['noise_floor']['sigma']:.2f} bpm/min (MAD sigma)")
    print(f"BETA                 {s['beta']} bpm/min  = {BETA_SIGMA} sigma\n")

    print(f"REAL  ({s['real']['hours']} h, {s['real']['dropout_pct']}% dropout, "
          f"{s['real']['abstain_pct']}% abstained -- gate assumed open)")
    print(f"  resting {s['real']['resting_bpm']} bpm, peak {s['real']['peak_bpm']}, "
          f"{s['real']['reserve_used_pct']}% of HR reserve")
    print(f"  {s['real']['episodes']} episodes, {s['real']['minutes_flagged']} min flagged, "
          f"load {s['real']['load']:,.0f} score-min")
    for _, e in er.nlargest(6, "area").sort_values("start").iterrows():
        print(f"    {e['start'].strftime('%H:%M:%S')}  {e['dur_min']:>5.2f} min  "
              f"peak {e['peak_score']:>5.1f}  ({e['peak_dHR_dt']:>5.1f} bpm/min, "
              f"+{e['hr_rise']:.1f} bpm)")

    w = s["walk"]
    print(f"\nSYNTHETIC WALK  ({w['steps']:,} steps, {w['walk_minutes']} min walking)")
    print(f"  precision {w['precision']}  recall {w['recall']}  F1 {w['f1']}"
          f"   (onset spans caught {w['onsets_hit']}/{w['onsets']})")
    print(f"  false positives while walking: {w['fp_while_walking']}   "
          f"on negative controls: {w['fp_negative_control']}")
    print(f"  {w['episodes']} episodes, load {w['load']:,.0f} score-min")
    print(f"  same data, gate removed: {w['episodes_ungated']} episodes, "
          f"load {w['load_ungated']:,.0f} score-min "
          f"({w['load_ungated']/max(w['load'],1):.1f}x inflation)")


if __name__ == "__main__":
    main()
