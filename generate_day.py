"""
Synthetic 8-hour Apple Watch + iPhone day for the bezel-MIT stress detector.

07:00:00 -> 14:59:50, 10-second resolution, 2880 samples.

He runs to work. Running raises cadence AND heart rate together, so the
movement gate closes on its own and the commute never reads as stress.
Then he sits down and the day happens to him.

Three layers, in order:
  1. HR_ANCHORS  - hand-placed heart-rate control points. All the taste is here.
  2. SEGMENTS    - hand-written step/activity spans + ground-truth labels.
  3. render()    - thin: PCHIP between anchors, AR(1) noise, respiratory
                   wobble, zero-inflated Poisson steps. No narrative decisions.
  4. GRACE_NOTES - hand-poked individual samples applied last.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

SEED = 137          # Sundai Hack 137
DAY = "2026-03-04"
T0, T1, DT = "07:00:00", "14:59:50", 10

# ---------------------------------------------------------------------------
# 1. HEART RATE ANCHORS  (time, bpm) -- every one of these placed by hand.
#    PCHIP interpolates between them; it will not overshoot, so a plateau
#    written as a plateau stays flat instead of ringing.
# ---------------------------------------------------------------------------
HR_ANCHORS = [
    # --- 07:00 waking up, moving around the apartment -----------------------
    ("07:00:00",  62), ("07:02:00",  60), ("07:04:30",  67), ("07:06:00",  61),
    ("07:09:00",  65), ("07:11:30",  59), ("07:13:00",  61),

    # --- 07:14 up and out the door, warm-up walk ----------------------------
    ("07:14:00",  66), ("07:15:30",  74), ("07:17:00",  81), ("07:19:00",  88),

    # --- 07:19 THE RUN. 33 minutes. -----------------------------------------
    ("07:21:00", 108), ("07:23:00", 126), ("07:25:00", 138), ("07:27:00", 146),
    ("07:30:00", 150), ("07:32:00", 152),
    ("07:34:00", 168), ("07:35:30", 166),          # the hill on the bridge
    ("07:37:00", 154), ("07:39:00", 151),
    # crosswalk. he stops dead; HR keeps climbing ~15s on cardiac lag.
    # this is a REAL micro-false-positive and it stays in on purpose.
    ("07:41:00", 154), ("07:41:40", 161),           # stopped, but HR still climbing
    ("07:42:00", 157), ("07:42:30", 147),
    ("07:42:50", 151),                              # green light, going again
    ("07:44:00", 156), ("07:46:00", 153), ("07:48:00", 158),
    ("07:50:00", 165), ("07:52:00", 172),           # last block, kicks it

    # --- 07:52 arrives. cooldown + stairs + HR recovery curve ---------------
    ("07:52:40", 170), ("07:53:30", 158),
    ("07:55:00", 146), ("07:55:40", 153),           # four flights of stairs
    ("07:57:00", 132), ("07:59:00", 118), ("08:02:00", 105),
    ("08:05:00",  94), ("08:09:00",  85), ("08:13:00",  79), ("08:17:00",  76),

    # --- 08:20 desk. calm morning. ------------------------------------------
    ("08:20:00",  74), ("08:26:00",  71), ("08:33:00",  70), ("08:40:00",  72),
    ("08:47:00",  69),
    ("08:55:00",  74), ("08:56:30",  82), ("08:58:30",  76),   # walks for coffee
    ("09:02:00",  71), ("09:10:00",  70), ("09:16:00",  71),
    ("09:22:00",  72), ("09:27:00",  71),

    # --- 09:30 STRESS #1. standup, and the news is bad. ---------------------
    ("09:30:00",  72), ("09:31:00",  84), ("09:32:00",  98), ("09:32:40", 102),
    ("09:34:00", 100), ("09:35:30", 103), ("09:37:00",  99), ("09:38:30",  92),
    ("09:40:00",  85), ("09:41:00",  82),

    # --- 09:41 back to work, but baseline sits ~6bpm high for an hour -------
    ("09:45:00",  80), ("09:52:00",  79), ("10:00:00",  81), ("10:08:00",  78),
    ("10:16:00",  80), ("10:24:00",  79), ("10:33:00",  81), ("10:40:00",  80),
    ("10:46:00",  79), ("10:50:00",  78),

    # --- 10:50 walks to the conference room ---------------------------------
    ("10:52:00",  88), ("10:54:00",  94), ("10:56:00",  96), ("10:57:00",  92),

    # --- 10:57 STRESS #2. the presentation. the long one. -------------------
    ("10:59:00",  90),
    ("11:01:00",  92), ("11:02:00",  98), ("11:03:00", 108),   # anticipatory ramp
    ("11:04:00", 116), ("11:05:00", 120),                      # he starts talking
    ("11:06:30", 124), ("11:08:00", 121), ("11:10:00", 126), ("11:12:00", 123),
    ("11:14:00", 128), ("11:16:00", 122), ("11:18:00", 119), ("11:20:00", 124),
    ("11:22:00", 121), ("11:25:00", 118),
    ("11:27:20", 119), ("11:28:00", 134), ("11:28:40", 141),   # "quick question--"
    ("11:29:30", 137), ("11:30:30", 130), ("11:32:00", 126), ("11:34:00", 122),
    ("11:36:00", 116), ("11:38:00", 110), ("11:40:00", 104), ("11:42:00",  99),
    ("11:45:00",  94),

    # --- 11:45 decompression ------------------------------------------------
    ("11:48:00",  88), ("11:52:00",  82), ("11:56:00",  78), ("12:00:00",  77),
    ("12:05:00",  76), ("12:10:00",  78),

    # --- 12:10 walks out for lunch ------------------------------------------
    ("12:12:00",  88), ("12:16:00",  92), ("12:20:00",  90), ("12:22:00",  80),

    # --- 12:22 eating. NEGATIVE CONTROL #1: postprandial drift, ~0.3 bpm/min.
    #     stationary and rising, but far too slow to clear beta.
    ("12:26:00",  76), ("12:30:00",  78), ("12:35:00",  80), ("12:40:00",  82),
    ("12:45:00",  83), ("12:50:00",  84),

    # --- 12:50 walks back ---------------------------------------------------
    ("12:52:00",  90), ("12:56:00",  92), ("13:00:00",  79),

    ("13:05:00",  77), ("13:10:00",  78), ("13:15:00",  76),

    # --- 13:20 NEGATIVE CONTROL #2: second coffee. ~0.6 bpm/min for 15 min. -
    ("13:20:00",  76), ("13:25:00",  80), ("13:30:00",  83), ("13:35:00",  85),

    # --- 13:35 small stressor. deliberately sits just ABOVE beta so it fires
    #     with a low score -- this is the one that tests threshold placement.
    ("13:36:20",  94), ("13:37:30",  97), ("13:39:00",  93), ("13:42:00",  87),
    ("13:47:00",  83), ("13:53:00",  81), ("14:00:00",  82), ("14:08:00",  79),

    # --- 14:15 STRESS #3. the email. sharpest dHR/dt of the day. ------------
    ("14:15:00",  80), ("14:15:40", 100), ("14:16:10", 118), ("14:16:40", 122),
    ("14:17:30", 117), ("14:18:30", 121), ("14:19:30", 119), ("14:21:00", 114),
    ("14:23:00", 109), ("14:25:00", 101), ("14:28:00",  93),

    # --- 14:28 winding down -------------------------------------------------
    ("14:32:00",  87), ("14:38:00",  83), ("14:44:00",  80),
    ("14:48:00",  84), ("14:50:00",  90), ("14:52:00",  86),   # walk to printer
    ("14:56:00",  80), ("14:59:50",  78),
]

# ---------------------------------------------------------------------------
# 2. SEGMENTS  (start, end, label, kind, steps_per_10s_bin)
#    `kind` becomes the ground-truth label. stress_onset = what the dHR/dt
#    detector actually claims to find; stress_sustained = elevated but flat,
#    which a derivative detector CANNOT see and is scored separately.
# ---------------------------------------------------------------------------
SEGMENTS = [
    ("07:00:00", "07:01:30", "still in bed, phone on the nightstand", "baseline",  0),
    ("07:01:30", "07:03:00", "up, bathroom",                          "baseline",  9),
    ("07:03:00", "07:05:30", "getting dressed",                       "baseline",  6),
    ("07:05:30", "07:07:00", "sitting on the bed, shoes",             "baseline",  1),
    ("07:07:00", "07:09:30", "kitchen, coffee",                       "baseline",  7),
    ("07:09:30", "07:11:30", "standing at the counter drinking it",   "baseline",  0),
    ("07:11:30", "07:13:00", "packing the bag",                       "baseline",  5),
    ("07:13:00", "07:14:00", "keys, door",                            "baseline",  3),

    ("07:14:00", "07:19:00", "walk out to the street (warm-up)",      "activity", 17),
    ("07:19:00", "07:41:00", "RUN to work",                           "activity", 29),
    ("07:41:00", "07:42:30", "crosswalk, stopped dead (90 s)",        "activity", 0),
    ("07:42:30", "07:52:20", "RUN to work (cont.)",                   "activity", 29),
    ("07:52:20", "07:54:00", "arrives, walking it off",               "activity", 16),
    ("07:54:00", "07:56:00", "stairs, four flights",                  "activity", 13),
    ("07:56:00", "08:02:00", "across the floor to the desk",          "activity",  5),
    ("08:02:00", "08:20:00", "sat down -- HR recovery curve",         "recovery",  0),

    ("08:20:00", "08:55:00", "desk work",                             "baseline",  0),
    ("08:55:00", "08:58:30", "coffee walk",                           "activity", 17),
    ("08:58:30", "09:30:00", "desk work",                             "baseline",  0),

    ("09:30:00", "09:33:00", "E1 standup: the news is bad",       "stress_onset",     0),
    ("09:33:00", "09:38:00", "E1 standup: sustained",             "stress_sustained", 0),
    ("09:38:00", "09:41:00", "E1 standup: decay",                 "recovery",         0),
    ("09:41:00", "10:50:00", "desk work (baseline sits ~6bpm high)",  "baseline",  0),

    ("10:50:00", "10:57:00", "walk to the conference room",           "activity", 19),
    ("10:57:00", "11:01:00", "sat down, waiting to go on",            "baseline",  0),
    ("11:01:00", "11:05:30", "E2 presentation: anticipatory ramp","stress_onset",     0),
    ("11:05:30", "11:12:00", "E2 presentation: talking",          "stress_sustained", 0),
    ("11:12:00", "11:14:00", "E2 presentation: PACING while talking", "stress_sustained", 11),
    ("11:14:00", "11:27:20", "E2 presentation: talking",          "stress_sustained", 0),
    ("11:27:20", "11:29:00", "E2 presentation: the hard question","stress_onset",     0),
    ("11:29:00", "11:45:00", "E2 presentation: decay",            "recovery",         0),

    ("11:45:00", "12:10:00", "back at the desk, decompressing",        "baseline",  0),
    ("12:10:00", "12:22:00", "walk out for lunch",                     "activity", 19),
    ("12:22:00", "12:50:00", "eating -- postprandial drift",     "negative_control", 0),
    ("12:50:00", "13:00:00", "walk back",                              "activity", 19),
    ("13:00:00", "13:20:00", "desk work",                              "baseline",  0),
    ("13:20:00", "13:35:00", "second coffee -- caffeine drift",  "negative_control", 0),

    ("13:35:00", "13:38:00", "E3 small stressor (borderline)",   "stress_onset",     0),
    ("13:38:00", "13:42:00", "E3 decay",                         "recovery",         0),
    ("13:42:00", "14:15:00", "desk work",                              "baseline",  0),

    ("14:15:00", "14:17:00", "E4 the deadline email",            "stress_onset",     0),
    ("14:17:00", "14:21:00", "E4 sustained",                     "stress_sustained", 0),
    ("14:21:00", "14:28:00", "E4 decay",                         "recovery",         0),

    ("14:28:00", "14:48:00", "desk work",                              "baseline",  0),
    ("14:48:00", "14:52:00", "walk to the printer",                    "activity", 18),
    ("14:52:00", "15:00:00", "desk work, end of window",               "baseline",  0),
]

# ---------------------------------------------------------------------------
# 3. GRACE NOTES -- individual samples poked by hand, applied after render.
#    ("add", n) offsets bpm; ("set", n) forces it; ("nan", None) is a dropout.
# ---------------------------------------------------------------------------
HR_POKES = {
    "07:33:10": ("add",  5),   # digs in on the steep part of the bridge
    "07:47:20": ("add",  4),   # swerves around a loose dog
    "09:14:10": ("add",  9),   # sneeze
    "09:14:20": ("add",  6),
    "09:14:30": ("add",  2),
    "10:33:20": ("nan", None), # band worked loose -- watch drops HR for 50s
    "10:33:30": ("nan", None),
    "10:33:40": ("nan", None),
    "10:33:50": ("nan", None),
    "10:34:00": ("nan", None),
    "11:12:40": ("add",  7),   # gesturing with the watch hand; PPG motion artifact
    "11:12:50": ("add",  5),
    "13:03:10": ("add", -6),   # bad PPG frame, undercounts
    "14:16:00": ("add",  3),   # E4 is jagged, not a smooth curve
    "14:18:00": ("add", -4),
    "14:19:00": ("add",  3),
    "14:41:10": ("nan", None), # second dropout, shorter
    "14:41:20": ("nan", None),
    "10:05:30": ("add",  4),   # phone rings, he sees who it is, answers anyway
    "10:05:40": ("add",  5),
    "10:05:50": ("add",  3),
    "08:44:20": ("add", -5),   # leans on the armrest, watch loses contact briefly
    "12:14:10": ("add",  4),   # jogs the last few steps across the crosswalk
    "13:11:00": ("add", -3),
    "11:33:20": ("add",  6),   # someone catches him on the way out to ask more
    "14:53:40": ("add",  4),
}

STEP_POKES = {
    "07:33:10": 31,   # surge up the hill
    "07:55:20": 22,   # stairs two at a time
    "08:12:20":  1,   # phone jostles as he leans back
    "08:31:40":  2,
    "09:14:00":  1,   # the sneeze registers as a step. it does. every time.
    "10:12:30":  3,   # leans over for the charger
    "11:19:20":  2,   # weight shift, mid-sentence
    "12:33:40":  4,   # reaching across the table
    "12:47:50":  2,
    "14:22:10":  2,
    "10:05:30":  1,   # stands up to take the call
    "10:05:40":  6,
    "10:05:50":  5,
    "10:06:00":  2,
    "11:33:20":  4,   # half-turns toward the door
    "13:26:40":  1,
    "14:35:10":  3,   # someone stops by the desk
    "14:35:20":  2,
}

# ---------------------------------------------------------------------------
# 4. RENDERER -- mechanical. Nothing above this line is decided here.
# ---------------------------------------------------------------------------

def _sec(hhmmss):
    h, m, s = (int(x) for x in hhmmss.split(":"))
    return h * 3600 + m * 60 + s


def build_grid():
    return np.arange(_sec(T0), _sec(T1) + DT, DT)


def render_hr(grid, rng):
    """PCHIP through the hand-placed anchors, plus two-timescale AR(1).

    Note on noise: at 10 s sampling the Nyquist limit is 0.05 Hz, so
    respiratory sinus arrhythmia (~0.25 Hz) and Mayer waves (~0.1 Hz) are
    both unresolvable -- modelling them here would just alias into garbage.
    What IS visible at this cadence is (a) fast beat-estimation noise and
    (b) slow autonomic drift, so that is what gets modelled: a fast AR(1)
    and a slow one, summed.
    """
    ax = np.array([_sec(t) for t, _ in HR_ANCHORS], float)
    ay = np.array([v for _, v in HR_ANCHORS], float)
    assert np.all(np.diff(ax) > 0), "HR anchors must be strictly increasing in time"
    base = PchipInterpolator(ax, ay)(grid)

    n = len(grid)
    fast = np.zeros(n)
    slow = np.zeros(n)
    phi_f, sd_f = 0.40, 1.00
    phi_s, sd_s = 0.97, 1.60
    for i in range(1, n):
        fast[i] = phi_f * fast[i - 1] + rng.normal(0, sd_f * np.sqrt(1 - phi_f ** 2))
        slow[i] = phi_s * slow[i - 1] + rng.normal(0, sd_s * np.sqrt(1 - phi_s ** 2))

    # measurement noise shrinks a little at high HR (stronger PPG signal)
    gain = np.clip(1.25 - 0.004 * base, 0.55, 1.15)
    return base + gain * (fast + slow)


def render_steps(grid, rng):
    """Zero-inflated when stationary; tight Gaussian when locomoting.

    Poisson is wrong for gait -- Poisson(29) is 174 +/- 32 spm, which no
    human produces. Real cadence is metronomic, so locomotion uses a narrow
    Gaussian and only the idle state gets the zero-inflated treatment.
    """
    steps = np.zeros(len(grid), int)
    for start, end, _lbl, _kind, c in SEGMENTS:
        m = (grid >= _sec(start)) & (grid < _sec(end))
        k = int(m.sum())
        if k == 0:
            continue
        if c == 0:
            # sitting: almost always 0, occasionally 1-2 from a hand movement
            hit = rng.random(k) < 0.035
            steps[m] = np.where(hit, rng.integers(1, 3, k), 0)
        elif c <= 8:
            # puttering around a room -- genuinely ragged, Poisson fits
            steps[m] = rng.poisson(c, k)
        else:
            # cap at 32/bin = 192 spm; above that is not a running cadence
            sd = max(0.7, 0.04 * c)
            steps[m] = np.clip(np.round(rng.normal(c, sd, k)), 0, 32).astype(int)
    return steps


def apply_grace_notes(grid, hr, steps):
    idx = {t: i for i, t in enumerate(grid)}
    for t, (mode, val) in HR_POKES.items():
        i = idx[_sec(t)]
        if mode == "nan":
            hr[i] = np.nan
        elif mode == "add":
            hr[i] += val
        else:
            hr[i] = val
    for t, val in STEP_POKES.items():
        steps[idx[_sec(t)]] = val
    return hr, steps


def main():
    rng = np.random.default_rng(SEED)
    grid = build_grid()
    hr = render_hr(grid, rng)
    steps = render_steps(grid, rng)
    hr, steps = apply_grace_notes(grid, hr, steps)

    ts = pd.to_datetime(DAY) + pd.to_timedelta(grid, unit="s")
    hr_r = np.round(hr, 1)

    pd.DataFrame({"timestamp": ts, "steps": steps}).to_csv(
        "data/steps_10s.csv", index=False)
    pd.DataFrame({"timestamp": ts, "hr_bpm": hr_r}).to_csv(
        "data/heart_rate_10s.csv", index=False)
    # the synced join -- signals only, no labels, so the detector cannot cheat
    pd.DataFrame({"timestamp": ts, "steps": steps, "hr_bpm": hr_r}).to_csv(
        "data/merged_10s.csv", index=False)

    gt = pd.DataFrame(
        [(f"{DAY} {s}", f"{DAY} {e}", lbl, kind) for s, e, lbl, kind, _ in SEGMENTS],
        columns=["start", "end", "label", "kind"])
    gt.to_csv("data/ground_truth_events.csv", index=False)

    # --- sanity gates ------------------------------------------------------
    assert len(grid) == 2880, f"expected 2880 samples, got {len(grid)}"
    assert np.all(np.diff(grid) == DT), "grid is not a uniform 10 s raster"
    assert steps.min() >= 0, "negative step count"
    fin = hr_r[~np.isnan(hr_r)]
    assert fin.min() > 40 and fin.max() < 200, f"HR out of range: {fin.min()}-{fin.max()}"
    covered = sum(_sec(e) - _sec(s) for s, e, *_ in SEGMENTS)
    assert covered == 8 * 3600, f"segments cover {covered}s, not 8h"

    print(f"{len(grid)} samples  {ts[0].time()} -> {ts[-1].time()}")
    print(f"steps: total {steps.sum():,}  max/10s {steps.max()}  "
          f"({steps.max()*6} spm peak cadence)")
    print(f"HR   : {np.nanmin(hr_r):.0f}-{np.nanmax(hr_r):.0f} bpm, "
          f"{int(np.isnan(hr_r).sum())} dropped samples")
    print(f"ground truth: {len(gt)} segments, "
          f"{(gt.kind=='stress_onset').sum()} onset spans")


if __name__ == "__main__":
    main()
