"""Assemble docs/ for GitHub Pages: a public home for the synthetic dataset
plus the interactive viewer. Run after generate_day.py / visualize.py /
build_viewer.py so everything published matches what is in data/.
"""
import pathlib, shutil, csv
import pandas as pd

DOCS = pathlib.Path("docs")
REPO = "https://github.com/aj872/bezel-MIT"

FILES = [
    ("steps_10s.csv", "iPhone step counts", "Pedometer count per 10-second bin.", True),
    ("heart_rate_10s.csv", "Apple Watch heart rate",
     "Blank cell where the watch dropped the sample — not interpolated.", True),
    ("merged_10s.csv", "Both, synced",
     "The join to feed a pipeline. Signals only, no labels, so a detector cannot cheat.", True),
    ("ground_truth_events.csv", "Ground truth",
     "The answer key. Keep it away from anything you are scoring.", False),
    ("stress_episodes.csv", "Detected episodes",
     "Output of stress_detect.py, not an input.", False),
    ("stress_timeline.csv", "Per-sample detector output",
     "Derivatives, gate state and score for every sample. Output, not an input.", False),
]

def preview(path, n=4):
    rows = list(csv.reader(open(path)))[:n+1]
    w = [max(len(r[i]) if i < len(r) else 0 for r in rows) for i in range(len(rows[0]))]
    return "\n".join("  ".join(c.ljust(w[i])[:44] for i, c in enumerate(r)) for r in rows)

def human(nbytes):
    return f"{nbytes/1024:.0f} KB" if nbytes < 1024**2 else f"{nbytes/1024**2:.1f} MB"

def main():
    if DOCS.exists():
        # docs/real/ is a second, independent site built by realviz/build.py.
        # Wipe only what this builder owns, or a rebuild here silently deletes it.
        for child in DOCS.iterdir():
            if child.name == "real":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    (DOCS / "data").mkdir(parents=True, exist_ok=True)
    (DOCS / "figures").mkdir(exist_ok=True)
    (DOCS / ".nojekyll").write_text("")

    for name, *_ in FILES:
        shutil.copy(f"data/{name}", DOCS / "data" / name)
    for p in pathlib.Path("figures").glob("*.png"):
        shutil.copy(p, DOCS / "figures" / p.name)
    shutil.copy("viewer/stress_viewer.html", DOCS / "viewer.html")

    cards = []
    for name, title, note, primary in FILES:
        src = pathlib.Path("data") / name
        rows = sum(1 for _ in open(src)) - 1
        cards.append(f'''
      <article class="f{' pri' if primary else ''}">
        <div class="fh">
          <div>
            <h3>{title}</h3>
            <code>data/{name}</code>
          </div>
          <a class="dl" href="data/{name}" download>Download</a>
        </div>
        <p class="meta">{rows:,} rows &middot; {human(src.stat().st_size)}</p>
        <p class="note">{note}</p>
        <pre>{preview(src)}</pre>
      </article>''')

    df = pd.read_csv("data/merged_10s.csv", parse_dates=["timestamp"])
    stats = dict(n=len(df), t0=df.timestamp.iloc[0].strftime("%H:%M:%S"),
                 t1=df.timestamp.iloc[-1].strftime("%H:%M:%S"),
                 steps=int(df.steps.sum()), drop=int(df.hr_bpm.isna().sum()),
                 hrlo=df.hr_bpm.min(), hrhi=df.hr_bpm.max())

    (DOCS / "index.html").write_text(PAGE.format(cards="".join(cards), repo=REPO, **stats))
    print(f"docs/  index.html + viewer.html + {len(FILES)} CSVs + figures")

PAGE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>bezel — synthetic validation set</title>
<meta name="description" content="Eight hours of 10-second iPhone step and Apple Watch heart-rate data with labelled stress events.">
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><text y='14' font-size='14'>&#128147;</text></svg>">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@600;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap">
<style>
:root{{
  color-scheme:light;
  --bg:#f2f4f6; --panel:#fbfcfd; --line:#dde3e9; --line-soft:#e9eef2;
  --ink:#0f1519; --ink-2:#4a5964; --ink-3:#8697a3;
  --accent:#eb6834; --blue:#2a78d6; --hr:#e34948;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  color-scheme:dark;
  --bg:#0d1114; --panel:#141a1f; --line:#26313a; --line-soft:#1d252c;
  --ink:#eef3f6; --ink-2:#9fb0bc; --ink-3:#687986;
  --accent:#d95926; --blue:#3987e5; --hr:#e66767;
}}}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
  font:400 15.5px/1.6 "IBM Plex Sans",system-ui,sans-serif;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1080px;margin:0 auto;padding:44px 22px 80px}}
h1{{font:700 clamp(30px,5vw,44px)/1.05 "IBM Plex Sans Condensed",sans-serif;
  margin:0 0 10px;letter-spacing:-.02em}}
.lede{{color:var(--ink-2);margin:0 0 26px;max-width:66ch;font-size:16.5px}}
.cta{{display:flex;flex-wrap:wrap;gap:10px;margin:0 0 34px}}
.cta a{{font:600 13.5px/1 "IBM Plex Sans",sans-serif;text-decoration:none;
  padding:12px 18px;border-radius:3px;border:1px solid var(--line);color:var(--ink-2);
  background:var(--panel);transition:.13s}}
.cta a:hover{{border-color:var(--ink-3);color:var(--ink)}}
.cta a.primary{{background:var(--accent);border-color:var(--accent);color:#fff}}
.cta a.primary:hover{{filter:brightness(1.08);color:#fff}}
.stats{{display:grid;gap:1px;background:var(--line);border:1px solid var(--line);
  grid-template-columns:repeat(auto-fit,minmax(132px,1fr));border-radius:3px;
  overflow:hidden;margin:0 0 40px}}
.stats div{{background:var(--panel);padding:13px 15px}}
.stats .k{{font:500 10.5px/1 "IBM Plex Mono",monospace;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-3)}}
.stats .v{{font:600 24px/1.05 "IBM Plex Sans Condensed",sans-serif;margin-top:7px;
  font-variant-numeric:tabular-nums}}
h2{{font:600 19px/1.2 "IBM Plex Sans Condensed",sans-serif;margin:0 0 6px;
  letter-spacing:-.005em}}
.sec{{margin:0 0 42px}}
.sec > p{{color:var(--ink-2);margin:0 0 18px;max-width:66ch;font-size:14.5px}}
.files{{display:grid;gap:14px}}
.f{{background:var(--panel);border:1px solid var(--line);border-radius:3px;padding:15px 17px 6px}}
.f.pri{{border-left:2px solid var(--accent)}}
.fh{{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}}
.f h3{{font:600 15px/1.2 "IBM Plex Sans",sans-serif;margin:0 0 3px}}
.f code{{font:500 12px "IBM Plex Mono",monospace;color:var(--ink-3)}}
.dl{{font:600 12px/1 "IBM Plex Sans",sans-serif;text-decoration:none;flex:none;
  padding:8px 13px;border:1px solid var(--line);border-radius:3px;color:var(--ink-2);
  background:var(--bg);transition:.13s}}
.dl:hover{{border-color:var(--accent);color:var(--accent)}}
.meta{{font:500 12px "IBM Plex Mono",monospace;color:var(--ink-3);margin:9px 0 0;
  font-variant-numeric:tabular-nums}}
.note{{color:var(--ink-2);font-size:13.5px;margin:5px 0 0}}
pre{{background:var(--bg);border:1px solid var(--line-soft);border-radius:3px;
  padding:11px 13px;margin:12px 0 15px;overflow-x:auto;
  font:400 11.5px/1.65 "IBM Plex Mono",monospace;color:var(--ink-2)}}
figure{{margin:0 0 18px}}
figure img{{width:100%;max-width:100%;display:block;border:1px solid var(--line);border-radius:3px}}
figcaption{{color:var(--ink-3);font-size:13px;margin-top:8px}}
.rule{{font:500 13.5px/1.85 "IBM Plex Mono",monospace;background:var(--panel);
  border:1px solid var(--line);border-radius:3px;padding:15px 17px;margin:0 0 18px;
  color:var(--ink-2);overflow-x:auto}}
.rule b{{color:var(--ink);font-weight:600}}
footer{{border-top:1px solid var(--line);margin-top:44px;padding-top:18px;
  color:var(--ink-3);font-size:13px}}
footer a{{color:var(--ink-2)}}
a{{color:var(--accent)}}
</style>
</head>
<body>
<div class="wrap">
  <h1>bezel</h1>
  <p class="lede">Eight hours of iPhone step counts and Apple Watch heart rate at 10-second
  resolution, with labelled stress events. Synthetic, seeded, and reproducible — built so a
  stress detector can be scored against a known answer key.</p>

  <div class="cta">
    <a class="primary" href="viewer.html">Open the interactive viewer</a>
    <a href="data/merged_10s.csv" download>Download the merged CSV</a>
    <a href="real/">The same detector on real data &rarr;</a>
    <a href="{repo}">Source on GitHub</a>
  </div>

  <div class="stats">
    <div><div class="k">samples</div><div class="v">{n:,}</div></div>
    <div><div class="k">duration</div><div class="v">8 h</div></div>
    <div><div class="k">resolution</div><div class="v">10s</div></div>
    <div><div class="k">total steps</div><div class="v">{steps:,}</div></div>
    <div><div class="k">HR range</div><div class="v">{hrlo:.0f}&ndash;{hrhi:.0f}</div></div>
    <div><div class="k">dropouts</div><div class="v">{drop}</div></div>
  </div>

  <section class="sec">
    <h2>The rule it exists to test</h2>
    <p>S is cumulative steps, so dS/dt is cadence. Heart rate that climbs while the step
    channel says you are still cannot be explained by movement.</p>
    <div class="rule">
      if  <b>dS/dt &lt; &alpha;</b>   you are not moving<br>
      and <b>dHR/dt &gt; &beta;</b>   but HR is climbing anyway<br>
      &rarr; movement cannot explain it<br>
      &rarr; <b>stress score &prop; dHR/dt</b>
    </div>
  </section>

  <section class="sec">
    <h2>Files</h2>
    <p>Timestamps run {t0} to {t1} on a uniform 10-second raster with no gaps in the grid
    itself. Regenerate any of these with <code>python generate_day.py</code> &mdash; it is
    seeded, so it reproduces byte-identically.</p>
    <div class="files">{cards}
    </div>
  </section>

  <section class="sec">
    <h2>Why the step channel is load-bearing</h2>
    <figure>
      <img src="figures/why_the_gate_matters.png" alt="Two windows with comparable dHR/dt: a run and a stressful meeting.">
      <figcaption>Remove the movement gate and the run commute alone contributes 41
      false positives peaking at 11.9 bpm/min &mdash; as steep as the meeting that is
      genuinely stressful.</figcaption>
    </figure>
    <figure>
      <img src="figures/stress_day_overview.png" alt="Full day: cadence, heart rate, dHR/dt and stress score.">
      <figcaption>The whole day. Cadence, heart rate, dHR/dt against &beta;, and the
      resulting stress score.</figcaption>
    </figure>
  </section>

  <footer>
    Sundai Hack 137 &middot; <a href="{repo}">aj872/bezel-MIT</a>
  </footer>
</div>
</body>
</html>
'''

if __name__ == "__main__":
    main()
