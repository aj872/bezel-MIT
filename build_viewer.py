"""Emit viewer/data.js from the detector, then splice it into a
self-contained viewer/stress_viewer.html.

The page recomputes the gate and the score in JavaScript so the alpha/beta
sliders are live; smoothing and both derivatives are precomputed here, since
they do not depend on either threshold. The JS detector was cross-checked
against stress_detect.py -- identical firing indices, identical stress load.
"""
import json, pathlib
import pandas as pd
import stress_detect as sd


def emit_data():
    df = sd.detect(sd.prepare(sd.load()))
    gt = pd.read_csv("data/ground_truth_events.csv", parse_dates=["start", "end"])
    t0 = df.timestamp.iloc[0]
    rnd = lambda s, n: [None if pd.isna(v) else round(float(v), n) for v in s]
    idx = lambda ts: int((ts - t0).total_seconds() // 10)

    payload = {
        "t0": "07:00:00", "dt": 10, "n": len(df),
        "steps": [int(v) for v in df.steps],
        "hr":    rnd(df.hr_bpm, 1),
        "hrs":   rnd(df.hr_smooth.where(df.valid), 2),
        "dhr":   rnd(df.dHR_dt, 3),
        "cad":   rnd(df.cadence_spm, 2),
        "valid": [int(v) for v in df.valid],
        "events": [{"s": idx(r.start), "e": idx(r.end), "k": r.kind, "l": r.label}
                   for _, r in gt.iterrows()],
        "defaults": {"alpha": sd.ALPHA, "beta": sd.BETA,
                     "scale": sd.SCALE, "debounce": sd.DEBOUNCE},
    }
    js = "window.DAY=" + json.dumps(payload, separators=(",", ":")) + ";"
    pathlib.Path("viewer/data.js").write_text(js)
    return len(js)


def main():
    n = emit_data()
    tpl = pathlib.Path("viewer/_template.html").read_text()
    data = pathlib.Path("viewer/data.js").read_text()
    out = tpl.replace("/*__DATA__*/", data)
    pathlib.Path("viewer/stress_viewer.html").write_text(out)
    print(f"viewer/data.js {n/1024:.0f} KB  ->  "
          f"viewer/stress_viewer.html {len(out)/1024:.0f} KB")


if __name__ == "__main__":
    main()
