-- One row per heart rate sample received from Pulsoid.
CREATE TABLE IF NOT EXISTS hr_sample (
  id           INTEGER PRIMARY KEY,
  source       TEXT    NOT NULL,          -- 'pulsoid'
  session_id   TEXT    NOT NULL,          -- one per socket lifetime
  bpm          INTEGER NOT NULL,
  measured_at  INTEGER NOT NULL,          -- unix ms, DEVICE clock
  received_at  INTEGER NOT NULL,          -- unix ms, OUR clock
  skew_ms      INTEGER NOT NULL,          -- received_at - measured_at
  UNIQUE (source, measured_at)
);

CREATE INDEX IF NOT EXISTS idx_hr_measured ON hr_sample (measured_at);
CREATE INDEX IF NOT EXISTS idx_hr_session  ON hr_sample (session_id, measured_at);

-- Gaps are first-class rows, not absences.
-- Downstream scoring reads this to decide whether it may emit a value.
CREATE TABLE IF NOT EXISTS hr_gap (
  id          INTEGER PRIMARY KEY,
  session_id  TEXT    NOT NULL,
  started_at  INTEGER NOT NULL,
  ended_at    INTEGER,                    -- NULL = still open
  reason      TEXT    NOT NULL            -- 'offline' | 'socket_close' | 'shutdown'
);

CREATE INDEX IF NOT EXISTS idx_gap_session ON hr_gap (session_id, started_at);

-- One row per socket lifetime. Lets you segment data by connection quality.
CREATE TABLE IF NOT EXISTS ingest_session (
  session_id  TEXT PRIMARY KEY,
  started_at  INTEGER NOT NULL,
  ended_at    INTEGER,
  reconnects  INTEGER NOT NULL DEFAULT 0
);