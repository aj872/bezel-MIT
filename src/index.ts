import 'dotenv/config';
import { openDb } from './db/index.js';
import { startIngest } from './ingest.js';

const token = process.env.PULSOID_TOKEN;
const dbPath = process.env.DB_PATH ?? './data/hr.db';

if (!token) {
  console.error('PULSOID_TOKEN is not set. Copy .env.example to .env and fill it in.');
  process.exit(1);
}

const db = openDb(dbPath);
const ingest = startIngest(db, token);

console.log(`session ${ingest.sessionId}`);
console.log(`writing to ${dbPath}`);
console.log('ctrl-c to stop\n');

let shuttingDown = false;
function shutdown(): void {
  if (shuttingDown) return;
  shuttingDown = true;

  const stats = ingest.stop();
  db.close();

  console.log('\n\n========== SESSION SUMMARY ==========');
  console.log(`session      ${stats.sessionId}`);
  console.log(`samples      ${stats.samples}`);
  console.log(`duplicates   ${stats.duplicates}`);
  console.log(`gaps         ${stats.gaps}`);
  console.log(`reconnects   ${stats.reconnects}`);
  console.log('=====================================');
  process.exit(0);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);