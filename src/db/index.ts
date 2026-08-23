import Database from 'better-sqlite3';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));

export type GapReason = 'offline' | 'socket_close' | 'shutdown';

export interface HrSample {
  source: string;
  sessionId: string;
  bpm: number;
  measuredAt: number;
  receivedAt: number;
}

export function openDb(path: string) {
  const db = new Database(path);

  // WAL lets a reader query the file while ingestion is still writing.
  db.pragma('journal_mode = WAL');
  db.pragma('foreign_keys = ON');

  db.exec(readFileSync(join(here, 'schema.sql'), 'utf8'));

  const stmts = {
    insertSample: db.prepare(`
      INSERT OR IGNORE INTO hr_sample
        (source, session_id, bpm, measured_at, received_at, skew_ms)
      VALUES
        (@source, @sessionId, @bpm, @measuredAt, @receivedAt, @skewMs)
    `),
    startSession: db.prepare(`
      INSERT INTO ingest_session (session_id, started_at) VALUES (?, ?)
    `),
    endSession: db.prepare(`
      UPDATE ingest_session SET ended_at = ? WHERE session_id = ?
    `),
    bumpReconnects: db.prepare(`
      UPDATE ingest_session SET reconnects = reconnects + 1 WHERE session_id = ?
    `),
    openGap: db.prepare(`
      INSERT INTO hr_gap (session_id, started_at, reason) VALUES (?, ?, ?)
    `),
    closeGap: db.prepare(`
      UPDATE hr_gap SET ended_at = ? WHERE id = ?
    `),
    countSamples: db.prepare(`
      SELECT COUNT(*) AS n FROM hr_sample WHERE session_id = ?
    `),
  };

  return {
    raw: db,

    /** Returns true if the sample was new, false if it was a duplicate. */
    insertSample(s: HrSample): boolean {
      const info = stmts.insertSample.run({
        source: s.source,
        sessionId: s.sessionId,
        bpm: s.bpm,
        measuredAt: s.measuredAt,
        receivedAt: s.receivedAt,
        skewMs: s.receivedAt - s.measuredAt,
      });
      return info.changes > 0;
    },

    startSession(sessionId: string, at: number): void {
      stmts.startSession.run(sessionId, at);
    },

    endSession(sessionId: string, at: number): void {
      stmts.endSession.run(at, sessionId);
    },

    bumpReconnects(sessionId: string): void {
      stmts.bumpReconnects.run(sessionId);
    },

    /** Returns the new gap's row id so it can be closed later. */
    openGap(sessionId: string, startedAt: number, reason: GapReason): number {
      const info = stmts.openGap.run(sessionId, startedAt, reason);
      return Number(info.lastInsertRowid);
    },

    closeGap(gapId: number, endedAt: number): void {
      stmts.closeGap.run(endedAt, gapId);
    },

    sampleCount(sessionId: string): number {
      const row = stmts.countSamples.get(sessionId) as { n: number };
      return row.n;
    },

    close(): void {
      db.close();
    },
  };
}

export type Db = ReturnType<typeof openDb>;