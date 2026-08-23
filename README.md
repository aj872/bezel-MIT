# bezel-MIT

Literature seed for the Sundai Hack 137 (AI 4 Next Gen Medicine) project: find a wearable-data paper that predicts a real health outcome, and actually implement it end to end on an Apple Watch.

## `papers.csv`

Six seed entries. Columns:

| column | meaning |
|---|---|
| `paper_title` | paper / artifact title |
| `year`, `venue` | when and where |
| `data_modality` | what signal off the wrist, and from whom |
| `analysis_method` | model / analysis actually run |
| `medical_outcome_predicted` | the clinical endpoint |
| `reported_result` | headline metric, as a sentence (AUROC, sens/spec, accuracy, or a qualitative claim) |
| `data_availability` | OPEN / CLOSED — can you get the training data |
| `apple_watch_feasibility` | HIGH / MEDIUM / LOW — can you actually rebuild this on a consumer Apple Watch |
| `source_citation` | authors + link(s) |

## The one constraint that kills most of these

**Apple does not expose raw PPG waveforms to third-party apps.** Anything downstream of raw PPG (most of the AF literature, Apple's own PPG foundation model) is not reproducible on a retail watch without an external sensor.

What *is* reachable: raw accelerometer at ~50 Hz (`CMMotionManager`), heart rate, `heartRateVariabilitySDNN`, `restingHeartRate`, `respiratoryRate`, `oxygenSaturation`, sleep stages, and Apple's own derived metrics including `appleSleepingBreathingDisturbances` and AFib History burden.

## Current ranking for a one-day build

1. **Walch 2019 sleep staging** — open PhysioNet dataset (`sleep-accel`), Apple Watch accel + HR, PSG labels. Only row that is both OPEN data and HIGH feasibility.
2. **Esmaeilpour 2024 respiratory infection** — Fitbit in the paper, but all four inputs are HealthKit types. Anomaly detection on nocturnal RHR/RR/HRV, no labels needed to run the detector. Note the honest weakness: PPV 4–10%, most alerts are stress.
3. **Sleep apnea** — you can read Apple's shipped breathing-disturbance metric *and* try to beat it on open PSG data (MESA, SHHS).

## Local setup

Requires Node 22+.

1. `npm i`
2. `cp .env.example .env`
3. Add your Pulsoid token to `.env` (create one at https://pulsoid.net/ui/keys)
4. `npm run dev`

`better-sqlite3` is a native module. If you see a Node ABI mismatch,
run `npm rebuild better-sqlite3`.
