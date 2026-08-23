export interface PulsoidSample {
    bpm: number;
    measuredAt: number;   // unix ms, device clock
    receivedAt: number;   // unix ms, our clock
  }
  
  export interface PulsoidHandlers {
    onSample: (s: PulsoidSample) => void;
    onOnline: () => void;
    onOffline: () => void;
    onReconnect: (attempt: number) => void;
    onFatal: (reason: string) => void;
  }
  
  const WS_URL = 'wss://dev.pulsoid.net/api/v1/data/real_time';
  const OFFLINE_MS = 30_000;
  const MAX_ATTEMPTS = 60;
  
  export function connectPulsoid(
    token: string,
    h: PulsoidHandlers,
  ): { disconnect: () => void } {
    let ws: WebSocket | null = null;
    let offlineTimer: NodeJS.Timeout | null = null;
    let retryTimer: NodeJS.Timeout | null = null;
    let attempt = 0;
    let online = false;
    let stopped = false;
  
    function armOfflineTimer(): void {
      if (offlineTimer) clearTimeout(offlineTimer);
      offlineTimer = setTimeout(() => {
        if (online) {
          online = false;
          h.onOffline();
        }
      }, OFFLINE_MS);
    }
  
    function handleFrame(raw: string): void {
      const receivedAt = Date.now();
      let msg: unknown;
      try {
        msg = JSON.parse(raw);
      } catch {
        return; // unparseable, drop rather than store garbage
      }
  
      const m = msg as { measured_at?: unknown; data?: { heart_rate?: unknown } };
      const bpm = m?.data?.heart_rate;
      const measuredAt = m?.measured_at;
      if (typeof bpm !== 'number' || typeof measuredAt !== 'number') return;
  
      if (!online) {
        online = true;
        h.onOnline();
      }
      armOfflineTimer();
      h.onSample({ bpm, measuredAt, receivedAt });
    }
  
    function scheduleRetry(): void {
      if (stopped) return;
      if (attempt >= MAX_ATTEMPTS) {
        h.onFatal(`giving up after ${attempt} reconnect attempts`);
        return;
      }
      attempt += 1;
      const delay = Math.min(30_000, 1_000 * 2 ** Math.min(attempt, 5));
      h.onReconnect(attempt);
      retryTimer = setTimeout(connect, delay);
    }
  
    function connect(): void {
      if (stopped) return;
      ws = new WebSocket(`${WS_URL}?access_token=${encodeURIComponent(token)}`);
  
      ws.addEventListener('open', () => {
        attempt = 0;
        armOfflineTimer();
      });
  
      ws.addEventListener('message', (e: MessageEvent) => {
        handleFrame(typeof e.data === 'string' ? e.data : String(e.data));
      });
  
      ws.addEventListener('error', () => {
        // 'close' always follows; retry logic lives there.
      });
  
      ws.addEventListener('close', (e: CloseEvent) => {
        if (online) {
          online = false;
          h.onOffline();
        }
        // These codes usually mean a bad token. Don't retry forever.
        if (e.code === 1008 || e.code === 4001) {
          h.onFatal(`socket rejected (code ${e.code}) — check PULSOID_TOKEN`);
          return;
        }
        scheduleRetry();
      });
    }
  
    connect();
  
    return {
      disconnect: () => {
        stopped = true;
        if (offlineTimer) clearTimeout(offlineTimer);
        if (retryTimer) clearTimeout(retryTimer);
        try {
          ws?.close();
        } catch {
          /* already closed */
        }
      },
    };
  }