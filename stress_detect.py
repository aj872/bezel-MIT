"""
Stress detection from iPhone steps + Apple Watch heart rate.

    S = cumulative steps, so dS/dt IS cadence (steps/min).

    if  dS/dt < alpha   (you are not moving)
    and dHR/dt > beta   (your heart rate is climbing anyway)
    -> the HR change is not attributable to movement -> attribute it to stress
    stress_score  proportional to  dHR/dt

Both derivatives are taken over a 60 s window. At 10 s spacing a single-sample
diff has a noise floor around +/-20 bpm/min, which would swamp any beta worth
picking, so the raw diff is never used.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ALPHA = 10.0    # steps/min -- below this you are, for our purposes, sitting still
# beta is set from the measured noise floor, not by taste: baseline dHR/dt on
# this recording is -0.17 +/- 1.21 bpm/min, so 4.0 sits at ~3.3 sigma. Dropping
# to 3.0 buys recall but falls inside the 3-sigma band. See the sweep in README.
BETA  = 4.0     # bpm/min   -- above this your heart rate is meaningfully climbing
SCALE = 25.0    # bpm/min mapped to a stress score of 100
DEBOUNCE = 2    # consecutive windows to fire. Inert on this data (see README);
                # kept as cheap insurance against single-sample flicker.
MAX_GAP = 3     # bridge watch dropouts up to 30 s; longer = no decision

DT_MIN = 10 / 60.0


def load(path="data/merged_10s.csv"):
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def prepare(df):
    """Smooth, bridge short gaps, and take both derivatives over 60 s."""
    hr = df["hr_bpm"].astype(float)

    # a dropout longer than MAX_GAP samples is not interpolated -- the detector
    # abstains there rather than inventing a slope across missing data
    isna = hr.isna()
    grp = (isna != isna.shift()).cumsum()
    runlen = isna.groupby(grp).transform("size")
    df["valid"] = ~(isna & (runlen > MAX_GAP))

    hr_f = hr.interpolate(limit=MAX_GAP, limit_direction="both")
    hr_f = hr_f.rolling(3, center=True, min_periods=1).median()
    hr_f = hr_f.ewm(span=5, adjust=False).mean()
    # run the EMA backwards too, so the smoother introduces no phase lag --
    # otherwise every detected onset lands a few samples late
    hr_f = hr_f[::-1].ewm(span=5, adjust=False).mean()[::-1]
    df["hr_smooth"] = hr_f

    filled = hr_f.bfill().ffill().to_numpy()
    # Savitzky-Golay 1st derivative, 7 samples = 60 s, quadratic fit
    df["dHR_dt"] = savgol_filter(filled, 7, 2, deriv=1, delta=10.0) * 60.0

    # dS/dt: steps accumulated per minute, centered
    df["cadence_spm"] = (df["steps"].rolling(6, center=True, min_periods=1).sum()
                         / (6 * DT_MIN) * DT_MIN * 6)
    df["cadence_spm"] = df["steps"].rolling(6, center=True, min_periods=1).mean() * 6.0
    return df


def detect(df, alpha=ALPHA, beta=BETA, debounce=DEBOUNCE):
    still  = df["cadence_spm"] < alpha
    rising = df["dHR_dt"] > beta
    raw = still & rising & df["valid"]

    # require `debounce` consecutive windows. this is what kills the 07:41
    # crosswalk artifact, where he stops dead but HR lags upward for ~15 s.
    fired = raw.copy()
    for k in range(1, debounce):
        fired &= raw.shift(k, fill_value=False)
    # credit the whole sustained run, not just its tail
    for k in range(1, debounce):
        fired |= fired.shift(-k, fill_value=False)

    df["gate_open"] = raw
    df["is_stress"] = fired
    df["stress_score"] = np.where(
        fired, np.clip(df["dHR_dt"] / SCALE * 100.0, 0, 100), 0.0)
    return df


def episodes(df, merge_gap=6):
    """Contiguous firings, merged across gaps shorter than merge_gap samples."""
    idx = np.flatnonzero(df["is_stress"].to_numpy())
    if idx.size == 0:
        return pd.DataFrame(columns=["start", "end", "dur_min", "peak_score",
                                     "peak_dHR_dt", "area"])
    splits = np.flatnonzero(np.diff(idx) > merge_gap) + 1
    out = []
    for run in np.split(idx, splits):
        a, b = run[0], run[-1]
        seg = df.iloc[a:b + 1]
        out.append({
            "start": df["timestamp"].iloc[a],
            "end": df["timestamp"].iloc[b],
            "dur_min": round((b - a + 1) * DT_MIN, 2),
            "peak_score": round(seg["stress_score"].max(), 1),
            "peak_dHR_dt": round(seg["dHR_dt"].max(), 1),
            "area": round(seg["stress_score"].sum() * DT_MIN, 1),
        })
    return pd.DataFrame(out)


def label_samples(df, gt):
    kind = pd.Series(["baseline"] * len(df), index=df.index)
    for _, r in gt.iterrows():
        m = (df["timestamp"] >= r["start"]) & (df["timestamp"] < r["end"])
        kind[m] = r["kind"]
    return kind


def evaluate(df, gt):
    kind = label_samples(df, gt)
    df["gt_kind"] = kind
    pred = df["is_stress"].to_numpy()

    # stress_sustained is elevated-but-flat. a derivative detector structurally
    # cannot see it, so it is excluded from scoring and reported on its own.
    scored = (kind != "stress_sustained").to_numpy()
    truth = (kind == "stress_onset").to_numpy()

    tp = int((pred & truth & scored).sum())
    fp = int((pred & ~truth & scored).sum())
    fn = int((~pred & truth & scored).sum())
    tn = int((~pred & ~truth & scored).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec  = tp / (tp + fn) if tp + fn else float("nan")
    f1   = 2 * prec * rec / (prec + rec) if prec + rec else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=prec, recall=rec, f1=f1), kind


def main():
    df = detect(prepare(load()))
    gt = pd.read_csv("data/ground_truth_events.csv", parse_dates=["start", "end"])
    stats, kind = evaluate(df, gt)
    eps = episodes(df)

    df.to_csv("data/stress_timeline.csv", index=False)
    eps.to_csv("data/stress_episodes.csv", index=False)

    print(f"alpha={ALPHA:g} spm   beta={BETA:g} bpm/min   scale={SCALE:g}   "
          f"debounce={DEBOUNCE} ({DEBOUNCE*10}s)\n")

    print("EPISODES")
    for _, e in eps.iterrows():
        lbl = gt[(gt.start <= e["start"]) & (gt.end > e["start"])]
        name = lbl.iloc[0]["label"] if len(lbl) else "?"
        print(f"  {e['start'].time()}  {e['dur_min']:>5.2f} min  "
              f"peak {e['peak_score']:>5.1f}  ({e['peak_dHR_dt']:>5.1f} bpm/min)  {name}")

    print(f"\nstress load (integral of score) = {df['stress_score'].sum()*DT_MIN:,.0f} score-min")

    print("\nPER-SAMPLE, vs ground truth  (stress_sustained excluded -- see below)")
    print(f"  TP {stats['tp']:4d}   FP {stats['fp']:4d}   "
          f"FN {stats['fn']:4d}   TN {stats['tn']:4d}")
    print(f"  precision {stats['precision']:.3f}   recall {stats['recall']:.3f}   "
          f"F1 {stats['f1']:.3f}")

    print("\nFIRINGS BY GROUND-TRUTH SEGMENT KIND")
    for k, sub in df.groupby("gt_kind"):
        n = int(sub["is_stress"].sum())
        print(f"  {k:<18} {n:>4} / {len(sub):>4} samples fired")

    print("\nEVENT-LEVEL RECALL (did each onset span get caught at all?)")
    for _, r in gt[gt.kind == "stress_onset"].iterrows():
        m = (df["timestamp"] >= r["start"]) & (df["timestamp"] < r["end"])
        hit = bool(df.loc[m, "is_stress"].any())
        pk = df.loc[m, "dHR_dt"].max()
        print(f"  [{'HIT ' if hit else 'MISS'}] {r['start'].time()}  "
              f"peak dHR/dt {pk:5.1f}  {r['label']}")

    print("\nMUST STAY SILENT")
    for k in ("activity", "negative_control", "recovery"):
        m = df["gt_kind"] == k
        n = int(df.loc[m, "is_stress"].sum())
        print(f"  {k:<18} {n} firings" + ("  <-- FALSE POSITIVE" if n else "  ok"))


if __name__ == "__main__":
    main()
