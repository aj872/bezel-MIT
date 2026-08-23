"""Static figures for the synthetic stress day. Writes figures/*.png."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import stress_detect as sd

# validated against scripts/validate_palette.js -- all six checks PASS, both modes
C = dict(cadence="#2a78d6", hr="#e34948", dhr="#4a3aa7", stress="#eb6834",
         surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", ink3="#8a8880",
         grid="#e6e5e1")
BAND = {"stress_onset": ("#eb6834", 0.16), "stress_sustained": ("#eda100", 0.10),
        "activity": ("#2a78d6", 0.09), "negative_control": ("#1baf7a", 0.10),
        "recovery": ("#8a8880", 0.06)}

plt.rcParams.update({
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"],
    "font.family": "sans-serif", "font.size": 9,
    "axes.edgecolor": C["grid"], "axes.labelcolor": C["ink2"],
    "xtick.color": C["ink3"], "ytick.color": C["ink3"],
    "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": C["grid"], "grid.linewidth": 0.6,
})


def bands(ax, gt, kinds):
    for _, r in gt.iterrows():
        if r["kind"] in kinds:
            col, a = BAND[r["kind"]]
            ax.axvspan(r["start"], r["end"], color=col, alpha=a, lw=0, zorder=0)


def main():
    df = sd.detect(sd.prepare(sd.load()))
    gt = pd.read_csv("data/ground_truth_events.csv", parse_dates=["start", "end"])
    t = df["timestamp"]

    fig, axes = plt.subplots(4, 1, figsize=(15, 11.5), sharex=True,
                             gridspec_kw=dict(hspace=0.16, height_ratios=[1, 1.35, 1, 1]))

    # ---- 1. cadence -------------------------------------------------------
    ax = axes[0]
    bands(ax, gt, BAND)
    ax.fill_between(t, 0, df["cadence_spm"], color=C["cadence"], alpha=0.85, lw=0)
    ax.axhline(sd.ALPHA, color=C["ink2"], ls=(0, (5, 3)), lw=1.2, zorder=5)
    ax.text(t.iloc[8], sd.ALPHA + 12, f"α = {sd.ALPHA:g} steps/min", color=C["ink2"],
            fontsize=8.5, fontweight="bold", va="bottom",
            bbox=dict(fc=C["surface"], ec="none", pad=1.5))
    ax.set_ylabel("cadence  (steps/min)", fontweight="bold")
    ax.set_title("dS/dt  —  cadence from iPhone step counts",
                 loc="left", color=C["ink"], fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, 205)

    # ---- 2. heart rate ----------------------------------------------------
    ax = axes[1]
    bands(ax, gt, BAND)
    ax.plot(t, df["hr_bpm"], color=C["hr"], lw=0.8, alpha=0.38, zorder=2)
    hs = df["hr_smooth"].where(df["valid"])          # keep dropouts visibly broken
    ax.plot(t, hs, color=C["hr"], lw=1.9, zorder=3)
    for _, r in gt[gt.kind == "stress_onset"].iterrows():
        m = (t >= r["start"]) & (t < r["end"])
        ax.plot(t[m], hs[m], color=C["hr"], lw=3.4, zorder=4, solid_capstyle="round")
    pad = pd.Timedelta(seconds=45)
    for _, g in df[~df["valid"]].groupby((df["valid"] != df["valid"].shift()).cumsum()):
        a, b = g["timestamp"].iloc[0] - pad, g["timestamp"].iloc[-1] + pad
        ax.axvspan(a, b, facecolor="none", edgecolor=C["ink3"], hatch="////",
                   lw=0, alpha=0.55, zorder=1)
        ax.annotate("watch\ndropout", xy=(a + (b - a) / 2, 176), color=C["ink2"],
                    fontsize=7.5, ha="center", va="top", linespacing=0.95)
    ax.set_ylabel("heart rate  (bpm)", fontweight="bold")
    ax.set_title("Apple Watch heart rate  —  faint = raw 10 s samples, "
                 "bold = smoothed, hatched = watch dropout (detector abstains)",
                 loc="left", color=C["ink"], fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(48, 185)

    ann = [("07:25:00", 150, "runs to work"), ("08:05:00", 118, "HR recovery"),
           ("09:31:30", 112, "E1  standup"), ("11:05:00", 132, "E2  presentation"),
           ("11:28:00", 152, "hard question"), ("12:35:00", 96, "lunch"),
           ("14:16:20", 134, "E4  the email")]
    for tt, y, lab in ann:
        ax.annotate(lab, xy=(pd.Timestamp(f"2026-03-04 {tt}"), y),
                    color=C["ink"], fontsize=8.5, fontweight="bold", ha="center")

    # ---- 3. dHR/dt --------------------------------------------------------
    ax = axes[2]
    bands(ax, gt, BAND)
    ax.axhline(0, color=C["grid"], lw=1)
    ax.plot(t, df["dHR_dt"], color=C["dhr"], lw=1.5)
    ax.axhline(sd.BETA, color=C["ink2"], ls=(0, (5, 3)), lw=1.2, zorder=5)
    ax.text(t.iloc[8], sd.BETA + 2.0, f"β = {sd.BETA:g} bpm/min", color=C["ink2"],
            fontsize=8.5, fontweight="bold", va="bottom",
            bbox=dict(fc=C["surface"], ec="none", pad=1.5))
    moving = (df["dHR_dt"] > sd.BETA) & (df["cadence_spm"] >= sd.ALPHA)
    ax.fill_between(t, sd.BETA, df["dHR_dt"], where=moving,
                    color=C["cadence"], alpha=0.50, lw=0, zorder=3)
    ax.fill_between(t, sd.BETA, df["dHR_dt"], where=df["gate_open"],
                    color=C["stress"], alpha=0.65, lw=0, zorder=4)
    ax.annotate("HR climbing just as fast here —\nbut he is running, so the gate\n"
                "suppresses it (41 samples)",
                xy=(pd.Timestamp("2026-03-04 07:26:00"), 11.5),
                xytext=(pd.Timestamp("2026-03-04 08:12:00"), 19),
                color=C["cadence"], fontsize=8.5, fontweight="bold", ha="left",
                arrowprops=dict(arrowstyle="->", color=C["cadence"], lw=1.3,
                                connectionstyle="arc3,rad=-0.25"))
    ax.set_ylabel("dHR/dt  (bpm/min)", fontweight="bold")
    ax.set_title("dHR/dt over a 60 s window  —  orange = STRESS (still and climbing)   ·   "
                 "blue = climbing, but movement explains it",
                 loc="left", color=C["ink"], fontsize=11, fontweight="bold", pad=8)

    # ---- 4. stress score --------------------------------------------------
    ax = axes[3]
    bands(ax, gt, BAND)
    ax.fill_between(t, 0, df["stress_score"], color=C["stress"], alpha=0.9, lw=0)
    for _, e in sd.episodes(df).iterrows():
        ax.annotate(f"{e['peak_score']:.0f}",
                    xy=(e["start"] + (e["end"] - e["start"]) / 2, e["peak_score"] + 4),
                    color=C["ink"], fontsize=9, fontweight="bold", ha="center")
    ax.set_ylabel("stress score", fontweight="bold")
    ax.set_title("stress score  ∝  dHR/dt, gated on stillness",
                 loc="left", color=C["ink"], fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(0, 118)
    ax.set_xlabel("time of day", fontweight="bold")

    for a in axes:
        a.grid(axis="y", zorder=0)
        a.set_xlim(t.iloc[0], t.iloc[-1])
        a.xaxis.set_major_locator(mdates.HourLocator())
        a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        a.xaxis.set_minor_locator(mdates.MinuteLocator(byminute=[30]))

    handles = [plt.Rectangle((0, 0), 1, 1, color=BAND[k][0], alpha=BAND[k][1] * 2.6)
               for k in ("stress_onset", "stress_sustained", "activity",
                         "negative_control", "recovery")]
    fig.legend(handles, ["stress onset", "stress sustained", "moving",
                         "negative control", "recovery"],
               loc="upper right", bbox_to_anchor=(0.995, 0.988), ncol=5,
               frameon=False, fontsize=8.5, labelcolor=C["ink2"],
               title="ground truth", title_fontsize=8.5)

    fig.suptitle("Stress isolated from heart rate that movement cannot explain",
                 x=0.008, ha="left", y=0.985, fontsize=14.5,
                 fontweight="bold", color=C["ink"])
    fig.text(0.008, 0.962, "synthetic day, 07:00–15:00, 10 s resolution  ·  "
             "iPhone steps + Apple Watch HR  ·  2,880 samples",
             ha="left", fontsize=9.5, color=C["ink2"])
    fig.subplots_adjust(top=0.935, left=0.062, right=0.995, bottom=0.055)
    fig.savefig("figures/stress_day_overview.png", dpi=170)
    print("wrote figures/stress_day_overview.png")
    figure_two(df)


def figure_two(df):
    """The argument in one image: two moments the heart cannot tell apart,
    and the step channel can."""
    t = df["timestamp"]
    D = "2026-03-04 "
    cases = [("07:19:00", "07:33:30", "RUNNING to work",
              "cadence 138–180 spm  →  gate CLOSED  →  score 0"),
             ("09:29:00", "09:36:00", "SITTING in a standup",
              f"cadence < {sd.ALPHA:g} spm  →  gate OPEN  →  peak score 61")]

    fig, axes = plt.subplots(2, 2, figsize=(13, 7),
                             gridspec_kw=dict(hspace=0.30, wspace=0.13,
                                              height_ratios=[1, 1]))
    for col, (a, b, title, verdict) in enumerate(cases):
        m = (t >= pd.Timestamp(D + a)) & (t <= pd.Timestamp(D + b))
        sub, ts = df[m], t[m]
        stressy = col == 1
        accent = C["stress"] if stressy else C["cadence"]

        ax = axes[0][col]
        ax.plot(ts, sub["hr_smooth"], color=C["hr"], lw=2.2)
        ax.set_ylabel("heart rate (bpm)", fontweight="bold") if col == 0 else None
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=accent)
        ax.text(0, 1.14, verdict, transform=ax.transAxes, fontsize=9.5,
                color=C["ink2"], fontweight="bold")

        ax = axes[1][col]
        ax.axhline(0, color=C["grid"], lw=1)
        ax.plot(ts, sub["dHR_dt"], color=C["dhr"], lw=1.8)
        ax.axhline(sd.BETA, color=C["ink2"], ls=(0, (5, 3)), lw=1.2)
        ax.fill_between(ts, sd.BETA, sub["dHR_dt"],
                        where=sub["dHR_dt"] > sd.BETA, color=accent,
                        alpha=0.60, lw=0)
        ax.set_ylim(-9, 18)
        ax.set_xlabel("time of day", fontweight="bold")
        if col == 0:
            ax.set_ylabel("dHR/dt (bpm/min)", fontweight="bold")
            ax.text(0.02, 0.90, f"β = {sd.BETA:g}", transform=ax.transAxes,
                    fontsize=9, fontweight="bold", color=C["ink2"])
        peak = sub["dHR_dt"].max()
        ax.annotate(f"peaks at {peak:.1f} bpm/min",
                    xy=(0.97, 0.88), xycoords="axes fraction", ha="right",
                    fontsize=10, fontweight="bold", color=accent)

    for row in axes:
        for ax in row:
            ax.grid(axis="y")
            ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=3))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    for r in (0, 1):
        lo = min(ax.get_ylim()[0] for ax in axes[r])
        hi = max(ax.get_ylim()[1] for ax in axes[r])
        for ax in axes[r]:
            ax.set_ylim(lo, hi)
        axes[r][1].tick_params(labelleft=False)

    fig.suptitle("Heart rate alone cannot tell these apart. Steps can.",
                 x=0.006, ha="left", y=0.978, fontsize=15, fontweight="bold",
                 color=C["ink"])
    fig.text(0.006, 0.933, "Comparable rate of climb, opposite cause. Without the movement "
             "gate the run alone contributes 41 false-positive samples.",
             ha="left", fontsize=10, color=C["ink2"])
    fig.subplots_adjust(top=0.845, left=0.068, right=0.988, bottom=0.085)
    fig.savefig("figures/why_the_gate_matters.png", dpi=170)
    print("wrote figures/why_the_gate_matters.png")


if __name__ == "__main__":
    main()
