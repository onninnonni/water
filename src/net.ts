// Crew networking through the pikos.org WebSocket relay. The first player in a room is host and
// runs the authoritative simulation; others send their inputs and avatar state.
export type Role = 'solo' | 'host' | 'guest';
export interface NetMsg { t: string; [k: string]: unknown; }

export class Net {
  role: Role = 'solo';
  room = '';
  id = 0;
  peers = new Set<number>();
  private ws: WebSocket | null = null;
  onMsg: (m: NetMsg) => void = () => undefined;
  onChange: () => void = () => undefined;

  static newRoom() {
    const A = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';
    return Array.from({ length: 5 }, () => A[Math.floor(Math.random() * A.length)]).join('');
  }

  connect(room: string): Promise<void> {
    this.room = room;
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(`wss://pikos.org/onni/seas/ws/${room}`);
      this.ws = ws;
      let settled = false;
      ws.onmessage = (e) => {
        let m: NetMsg;
        try { m = JSON.parse(e.data as string); } catch { return; }
        if (m.t === 'hello') {
          this.role = m.role as Role; this.id = m.id as number; this.peers = new Set(m.peers as number[]);
          if (!settled) { settled = true; resolve(); }
          this.onChange(); return;
        }
        if (m.t === 'join') { this.peers.add(m.id as number); this.onChange(); }
        if (m.t === 'leave') { this.peers.delete(m.id as number); this.onChange(); }
        if (m.t === 'promote') { this.role = 'host'; this.onChange(); }
        this.onMsg(m);
      };
      ws.onerror = () => { if (!settled) { settled = true; reject(new Error('Could not reach the crew server')); } };
      ws.onclose = () => { this.peers.clear(); if (this.role !== 'solo') { this.role = 'solo'; this.onChange(); } };
    });
  }
  send(m: NetMsg) { if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify({ ...m, from: this.id })); }
  get online() { return this.ws !== null && this.ws.readyState === 1; }
}
