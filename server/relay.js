// Seven Seas crew relay: a dependency-free WebSocket relay. Each room holds up to 4 players;
// the first to join is host. Messages from one member are forwarded to the others unchanged.
// Control messages from the relay: {t:'hello', role, id, peers}, {t:'join', id}, {t:'leave', id}, {t:'promote'}.
'use strict';
const http = require('http');
const crypto = require('crypto');

const PORT = Number(process.env.PORT) || 8798;
const PREFIX = '/onni/seas/ws/';
const MAX_ROOM = 4, MAX_ROOMS = 300, MAX_MSG = 64 * 1024, MAX_RATE = 120; // messages per second
const rooms = new Map(); // code -> Set<Client>
let nextId = 1;

function frame(text) {
  const payload = Buffer.from(text, 'utf8');
  const len = payload.length;
  let head;
  if (len < 126) { head = Buffer.alloc(2); head[1] = len; }
  else if (len < 65536) { head = Buffer.alloc(4); head[1] = 126; head.writeUInt16BE(len, 2); }
  else { head = Buffer.alloc(10); head[1] = 127; head.writeBigUInt64BE(BigInt(len), 2); }
  head[0] = 0x81;
  return Buffer.concat([head, payload]);
}
const control = (op, data = Buffer.alloc(0)) => Buffer.concat([Buffer.from([0x80 | op, data.length]), data]);

class Client {
  constructor(sock, room) {
    this.sock = sock; this.room = room; this.id = nextId++; this.buf = Buffer.alloc(0);
    this.host = false; this.alive = true; this.count = 0; this.windowStart = Date.now();
  }
  send(obj) { if (!this.sock.destroyed) this.sock.write(frame(typeof obj === 'string' ? obj : JSON.stringify(obj))); }
  close() { try { this.sock.end(control(0x8)); } catch { /* ignore */ } this.sock.destroy(); }
}

function leave(c) {
  const set = rooms.get(c.room);
  if (!set || !set.has(c)) return;
  set.delete(c);
  for (const o of set) o.send({ t: 'leave', id: c.id });
  if (c.host && set.size) { const h = set.values().next().value; h.host = true; h.send({ t: 'promote' }); }
  if (!set.size) rooms.delete(c.room);
}

function onData(c, chunk) {
  c.buf = Buffer.concat([c.buf, chunk]);
  while (c.buf.length >= 2) {
    const b0 = c.buf[0], b1 = c.buf[1], op = b0 & 0x0f, masked = (b1 & 0x80) !== 0;
    let len = b1 & 0x7f, off = 2;
    if (len === 126) { if (c.buf.length < 4) return; len = c.buf.readUInt16BE(2); off = 4; }
    else if (len === 127) { if (c.buf.length < 10) return; len = Number(c.buf.readBigUInt64BE(2)); off = 10; }
    if (len > MAX_MSG) { c.close(); return; }
    const need = off + (masked ? 4 : 0) + len;
    if (c.buf.length < need) return;
    let data = c.buf.subarray(off + (masked ? 4 : 0), need);
    if (masked) {
      const mask = c.buf.subarray(off, off + 4), out = Buffer.alloc(len);
      for (let i = 0; i < len; i++) out[i] = data[i] ^ mask[i & 3];
      data = out;
    }
    c.buf = c.buf.subarray(need);
    if (op === 0x8) { c.close(); return; }
    if (op === 0x9) { c.sock.write(control(0xA, data)); continue; }
    if (op === 0xA) { c.alive = true; continue; }
    if (op !== 0x1) continue;
    const now = Date.now();
    if (now - c.windowStart > 1000) { c.windowStart = now; c.count = 0; }
    if (++c.count > MAX_RATE) continue;
    const text = data.toString('utf8');
    for (const o of rooms.get(c.room) || []) if (o !== c) o.send(text);
  }
}

const server = http.createServer((req, res) => {
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({ ok: true, rooms: rooms.size }));
});

server.on('upgrade', (req, sock) => {
  const url = req.url || '';
  const code = url.startsWith(PREFIX) ? url.slice(PREFIX.length).split('?')[0].toUpperCase() : '';
  const key = req.headers['sec-websocket-key'];
  if (!/^[A-Z0-9]{4,8}$/.test(code) || !key) { sock.end('HTTP/1.1 400 Bad Request\r\n\r\n'); return; }
  let set = rooms.get(code);
  if (!set && rooms.size >= MAX_ROOMS) { sock.end('HTTP/1.1 503 Service Unavailable\r\n\r\n'); return; }
  if (set && set.size >= MAX_ROOM) { sock.end('HTTP/1.1 409 Conflict\r\n\r\n'); return; }
  const accept = crypto.createHash('sha1').update(key + '258EAFA5-E914-47DA-95CA-C5AB0DC85B11').digest('base64');
  sock.write('HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ' + accept + '\r\n\r\n');
  sock.setNoDelay(true);
  if (!set) { set = new Set(); rooms.set(code, set); }
  const c = new Client(sock, code);
  c.host = set.size === 0;
  const peers = [...set].map((o) => o.id);
  set.add(c);
  c.send({ t: 'hello', role: c.host ? 'host' : 'guest', id: c.id, peers });
  for (const o of set) if (o !== c) o.send({ t: 'join', id: c.id });
  sock.on('data', (d) => onData(c, d));
  sock.on('close', () => leave(c));
  sock.on('error', () => leave(c));
});

setInterval(() => {
  for (const set of rooms.values()) for (const c of set) {
    if (!c.alive) { c.close(); leave(c); continue; }
    c.alive = false; c.sock.write(control(0x9));
  }
}, 25000);

server.listen(PORT, '127.0.0.1', () => console.log('seas relay on 127.0.0.1:' + PORT));
