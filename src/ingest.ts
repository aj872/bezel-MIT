import { randomUUID } from 'node:crypto';
import type { Db, GapReason } from './db/index.js';
import { connectPulsoid, type PulsoidSample } from './pulsoid.js';

const SOURCE = 'pulsoid';

export interface IngestStats {
  sessionId: string;
  samples: number;
  duplicates: number;
  gaps: number;
  reconnects: number;
}

export function startIngest(db: Db, token: string) {
  const sessionId = randomUUID();
  const startedAt = Date.now();
  db.startSession(sessionId, startedAt);

  let samples = 0;
  let duplicates = 0;
  let gaps = 0;
  let reconnects = 0;

  // Tracks the currently-open gap, if any. Null means we have coverage.
  let openGapId: number | null = null;
  let lastSampleAt: number | null = null;

  function beginGap(reason: GapReason): void {
    if (openGapId !== null) return; // already in a gap, don't nest
    // Backdate the gap to the last good sample so the window is honest.
    const startedAt = lastSampleAt ?? Date.now();
    openGapId = db.openGap(sessionId, startedAt, reason);
    gaps += 1;
    console.log(`\n[gap opened] reason=${reason}`);
  }

  function endGap(): void {
    if (openGapId === null) return;
    db.closeGap(openGapId, Date.now());
    console.log(`[gap closed]`);
    openGapId = null;
  }

  const conn = connectPulsoid(token, {
    onSample: (s: PulsoidSample) => {
      endGap(); // any sample means coverage resumed
      const isNew = db.insertSample({
        source: SOURCE,
        sessionId,
        bpm: s.bpm,
        measuredAt: s.measuredAt,
        receivedAt: s.receivedAt,
      });
      if (isNew) samples += 1;
      else duplicates += 1;

      lastSampleAt = s.receivedAt;

      const skew = s.receivedAt - s.measuredAt;
      process.stdout.write(
        `\r${String(s.bpm).padStart(3)} bpm | n=${samples} dup=${duplicates} ` +
        `gaps=${gaps} skew=${skew}ms    `,
      );
    },

    onOnline: () => {
      console.log('\n[online] receiving samples');
    },

    onOffline: () => {
      beginGap('offline');
    },

    onReconnect: (attempt) => {
      reconnects += 1;
      db.bumpReconnects(sessionId);
      beginGap('socket_close');
      console.log(`\n[reconnect] attempt ${attempt}`);
    },

    onFatal: (reason) => {
      console.error(`\n[fatal] ${reason}`);
      stop();
      process.exitCode = 1;
    },
  });

  function stop(): IngestStats {
    conn.disconnect();
    // An open gap at shutdown gets closed now — leaving ended_at NULL
    // forever would make it look like the gap never resolved.
    if (openGapId !== null) {
      db.closeGap(openGapId, Date.now());
      openGapId = null;
    }
    db.endSession(sessionId, Date.now());
    return { sessionId, samples, duplicates, gaps, reconnects };
  }

  return { sessionId, stop };
}