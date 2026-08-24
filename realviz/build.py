"""Assemble docs/real/ -- the real-recording viewer, as its own GitHub Page.

Deliberately separate from build_site.py, which owns docs/ for the synthetic
day. This writes only inside docs/real/ and never touches the other site.

The page recomputes the gate and the score in JavaScript so alpha/beta/debounce
stay live; smoothing and both derivatives are precomputed here, since neither
depends on a threshold. The JS detector is cross-checked against detect.py in
verify() below -- identical firing indices and identical stress load, or the
build fails.
"""
import json
import pathlib
import shutil
import numpy as np
import pandas as pd

import realviz.detect as det
import realviz.ingest as ingest

DOCS = pathlib.Path("docs/real")
SRC = pathlib.Path("data/real")
FILES = ["hr_10s.csv", "walk_10s.csv", "walk_events.csv",
         "timeline_real.csv", "timeline_walk.csv",
         "episodes_real.csv", "episodes_walk.csv", "summary.json"]


def rnd(s, n):
    return [None if pd.isna(v) else round(float(v), n) for v in s]


def jumps(real, walk, er, n):
    """Preset views: the whole thing, then the moments worth landing on."""
    out = [["Whole session", 0, n - 1]]
    t0 = real["timestamp"].iloc[0]
    idx = lambda ts: int((ts - t0).total_seconds() // det.GRID_S)

    top = er.nlargest(2, "area").sort_values("start")
    for _, e in top.iterrows():
        a, z = idx(e["start"]), idx(e["end"])
        pad = max(48, z - a)
        out.append([f"Real {e['start'].strftime('%H:%M')}",
                    max(0, a - pad), min(n - 1, z + pad)])

    # the longest walking bout: the case the real track cannot rule out
    walking = (walk["cadence_spm"] > 30).to_numpy()
    best, cur = (0, 0), None
    for i, v in enumerate(walking):
        if v and cur is None:
            cur = i
        elif not v and cur is not None:
            if i - cur > best[1] - best[0]:
                best = (cur, i)
            cur = None
    out.append(["Walking bout", max(0, best[0] - 30), min(n - 1, best[1] + 30)])
    return out


def payload():
    real, walk, gt, er, ew, summary = det.run()
    n = len(real)
    t0 = real["timestamp"].iloc[0]
    idx = lambda ts: int((ts - t0).total_seconds() // det.GRID_S)

    track = lambda df, cad: {
        "hr":  rnd(df["hr_bpm"], 1),
        "hrs": rnd(df["hr_smooth"].where(df["valid"]), 2),
        "dhr": rnd(df["dHR_dt"], 3),
        "cad": rnd(df["cadence_spm"], 1) if cad else None,
        "valid": [int(v) for v in df["valid"]],
    }
    R = track(real, False)
    R["art"] = rnd(real["hr_artifact"], 1)
    R["cad"] = None

    return {
        "t0": t0.strftime("%H:%M:%S"),
        "t0s": t0.hour * 3600 + t0.minute * 60 + t0.second,
        "date": t0.strftime("%A %-d %B %Y"),
        "dt": det.GRID_S, "n": n,
        "real": R,
        "walk": track(walk, True),
        "events": [{"s": idx(r.start), "e": idx(r.end), "k": r.kind, "l": r.label}
                   for _, r in gt.iterrows() if r.kind != "baseline"],
        "ctx": {"real": det.hr_context(real), "walk": det.hr_context(walk)},
        "defaults": {"alpha": det.ALPHA, "beta": summary["beta"],
                     "scale": det.SCALE, "debounce": det.DEBOUNCE},
        "jumps": jumps(real, walk, er, n),
        "files": FILES,
    }, summary, real, walk, er


def verify(pay, real, walk, summary):
    """Re-run the JS detector's logic in numpy and demand it match detect.py."""
    for name, df, gated in (("real", real, False), ("walk", walk, True)):
        T = pay[name]
        dhr = np.array([np.nan if v is None else v for v in T["dhr"]])
        valid = np.array(T["valid"], bool)
        still = (np.array([np.nan if v is None else v for v in T["cad"]])
                 < det.ALPHA) if gated else np.ones(len(dhr), bool)
        raw = valid & np.nan_to_num(still, nan=False).astype(bool) & (dhr > summary["beta"])
        fired = raw.copy()
        for k in range(1, det.DEBOUNCE):
            fired &= np.concatenate([np.zeros(k, bool), raw[:-k]])
        f = fired.copy()
        for k in range(1, det.DEBOUNCE):
            f |= np.concatenate([fired[k:], np.zeros(k, bool)])
        want = df["is_stress"].to_numpy()
        bad = int((f != want).sum())
        if bad:
            raise AssertionError(f"{name}: JS and Python disagree on {bad} samples")
    return True


def main():
    pay, summary, real, walk, er = payload()
    verify(pay, real, walk, summary)

    DOCS.mkdir(parents=True, exist_ok=True)
    for p in list(DOCS.glob("*.html")) + list(DOCS.glob("data/*")):
        p.unlink()
    (DOCS / "data").mkdir(exist_ok=True)
    (DOCS / ".nojekyll").write_text("")
    for f in FILES:
        shutil.copy(SRC / f, DOCS / "data" / f)

    nf = summary["noise_floor"]
    raw = ingest.read_pulsoid(sorted(pathlib.Path(".").glob("pulsoid-report-*.csv"))[-1])
    subs = {
        "__HOURS__": f"{summary['real']['hours']:.2f}",
        "__RAW__": f"{len(raw):,}",
        "__DATE__": pay["date"],
        "__BETA__": f"{summary['beta']:.2f}",
        "__NF_MED__": f"{nf['median']:+.2f}",
        "__NF_SIG__": f"{nf['sigma']:.2f}",
        "__BSIG__": f"{summary['beta_sigma']}",
        "__ARTIFACTS__": str(int(ingest.mask_harmonics(raw)["artifact"].sum())),
        "__ABSTAIN__": f"{summary['real']['abstain_pct']:.1f}",
        "__REAL_EPS__": str(summary["real"]["episodes"]),
    }

    js = "window.REAL=" + json.dumps(pay, separators=(",", ":")) + ";"
    html = pathlib.Path("realviz/_template.html").read_text()
    html = html.replace("/*__DATA__*/", js)
    for k, v in subs.items():
        html = html.replace(k, v)
    left = [k for k in subs if k in html]
    if left:
        raise AssertionError(f"unsubstituted placeholders: {left}")
    (DOCS / "index.html").write_text(html)

    print(f"data payload   {len(js)/1024:.0f} KB")
    print(f"page           {len(html)/1024:.0f} KB  -> {DOCS/'index.html'}")
    print(f"csv            {len(FILES)} files -> {DOCS/'data'}")
    print(f"js/py detector agree on all {pay['n']:,} samples, both tracks")


if __name__ == "__main__":
    main()
