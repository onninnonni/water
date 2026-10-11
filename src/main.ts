import * as S from './shaders';
import { Mat4, perspective, lookAt, multiply } from './mat4';
import { IC, SUN_DIR, WIND_DIR, clamp, mix, mix3, smoothstep, stormAt, norm3 } from './world';
import { uWA, uWB, updatePhases, sampleWater, newSample, sigWaveHeight } from './waves';
import { buildLand, buildShoreMap, buildBuoy, PILLARS, ARCH, LIGHTHOUSE, BUOYS, SHORE, BUOY_COLS } from './island';
import {
  Ship, shipAxes, buildShipMesh, buildSailMesh, buildCannonMesh, buildHarpoonMesh, buildBoltMesh, buildBallMesh, buildWheelMesh,
  MASTS, CANNONS, HELM, HARPOON, deckZ, moveOnDeck,
} from './ship';
import { buildAvatarParts, avatarPose, RZ, TR, AvatarParts } from './avatar';
import { MonsterSystem, Monster, MS, segRadius, buildSegmentMesh, buildHeadMesh } from './monsters';
import { Net, NetMsg } from './net';
import { SeaAudio } from './audio';
import type { V3 } from './mesh';

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const isTouch = navigator.maxTouchPoints > 0 && matchMedia('(pointer: coarse)').matches;
const isMobile = isTouch || /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
if (isTouch) document.body.classList.add('touch');

function fail(msg: string) {
  const e = $('err'); e.style.display = 'flex'; e.textContent = msg;
  $('start').style.display = 'none';
}

// ---------- WebGL ----------
const canvas = $<HTMLCanvasElement>('c');
const glc = canvas.getContext('webgl2', { antialias: true, alpha: false, powerPreference: 'high-performance' });
if (!glc) { fail('This game needs WebGL2, which this browser does not provide.'); throw new Error('no webgl2'); }
const G: WebGL2RenderingContext = glc;
G.getExtension('EXT_color_buffer_float');
G.getExtension('EXT_color_buffer_half_float');

function compile(type: number, src: string) {
  const s = G.createShader(type)!;
  G.shaderSource(s, src); G.compileShader(s);
  if (!G.getShaderParameter(s, G.COMPILE_STATUS)) throw new Error(G.getShaderInfoLog(s) ?? 'shader error');
  return s;
}
class Program {
  prog: WebGLProgram;
  u = new Map<string, WebGLUniformLocation>();
  constructor(vs: string, fs: string, attribs: string[]) {
    const p = G.createProgram()!;
    G.attachShader(p, compile(G.VERTEX_SHADER, vs)); G.attachShader(p, compile(G.FRAGMENT_SHADER, fs));
    attribs.forEach((a, i) => G.bindAttribLocation(p, i, a));
    G.linkProgram(p);
    if (!G.getProgramParameter(p, G.LINK_STATUS)) throw new Error(G.getProgramInfoLog(p) ?? 'link error');
    this.prog = p;
    const n = G.getProgramParameter(p, G.ACTIVE_UNIFORMS) as number;
    for (let i = 0; i < n; i++) {
      const name = G.getActiveUniform(p, i)!.name;
      this.u.set(name.replace(/\[0\]$/, ''), G.getUniformLocation(p, name)!);
    }
  }
  use() { G.useProgram(this.prog); return this; }
  f(n: string, v: number) { const l = this.u.get(n); if (l) G.uniform1f(l, v); return this; }
  i(n: string, v: number) { const l = this.u.get(n); if (l) G.uniform1i(l, v); return this; }
  v2(n: string, a: number, b: number) { const l = this.u.get(n); if (l) G.uniform2f(l, a, b); return this; }
  v3(n: string, v: ArrayLike<number>) { const l = this.u.get(n); if (l) G.uniform3f(l, v[0], v[1], v[2]); return this; }
  v4a(n: string, v: Float32Array) { const l = this.u.get(n); if (l) G.uniform4fv(l, v); return this; }
  v2a(n: string, v: Float32Array) { const l = this.u.get(n); if (l) G.uniform2fv(l, v); return this; }
  m4(n: string, m: Mat4) { const l = this.u.get(n); if (l) G.uniformMatrix4fv(l, false, m); return this; }
}

let pSky: Program, pOcean: Program, pSolid: Program, pSpray: Program, pRain: Program, pGlow: Program, pBolt: Program, pBeam: Program, pWake: Program, pShift: Program;
try {
  pSky = new Program(S.VS_QUAD, S.FS_SKY, ['aPos']);
  pOcean = new Program(S.VS_OCEAN, S.FS_OCEAN, ['aGrid']);
  pSolid = new Program(S.VS_SOLID, S.FS_SOLID, ['aPos', 'aCol']);
  pSpray = new Program(S.VS_SPRAY, S.FS_SPRAY, ['aP']);
  pRain = new Program(S.VS_RAIN, S.FS_RAIN, []);
  pGlow = new Program(S.VS_GLOW, S.FS_GLOW, ['aCenter', 'aCorner', 'aCol']);
  pBolt = new Program(S.VS_BOLT, S.FS_BOLT, ['aPos', 'aU']);
  pBeam = new Program(S.VS_BEAM, S.FS_BEAM, ['aP']);
  pWake = new Program(S.VS_QUAD, S.FS_WAKE, ['aPos']);
  pShift = new Program(S.VS_QUAD, S.FS_SHIFT, ['aPos']);
} catch (e) { fail('Could not start the graphics: ' + (e as Error).message); throw e; }

// ---------- Geometry ----------
interface Mesh { vao: WebGLVertexArrayObject; buf: WebGLBuffer; count: number; }
function vao(layout: number[], data: Float32Array | null, usage: number = G.STATIC_DRAW): Mesh {
  const v = G.createVertexArray()!; G.bindVertexArray(v);
  const b = G.createBuffer()!; G.bindBuffer(G.ARRAY_BUFFER, b);
  if (data) G.bufferData(G.ARRAY_BUFFER, data, usage);
  const per = layout.reduce((a, n) => a + n, 0);
  let off = 0;
  layout.forEach((n, i) => { G.enableVertexAttribArray(i); G.vertexAttribPointer(i, n, G.FLOAT, false, per * 4, off); off += n * 4; });
  G.bindVertexArray(null);
  return { vao: v, buf: b, count: data ? data.length / per : 0 };
}
const quad = vao([2], new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]));
const solid = (d: number[]) => vao([3, 3], new Float32Array(d));

const ocean = (() => {
  const NR = isMobile ? 120 : 170, NS = isMobile ? 144 : 200, RMAX = 7000, a = 6, ea = Math.exp(a) - 1;
  const verts = new Float32Array((NR + 1) * NS * 3);
  let k = 0;
  for (let i = 0; i <= NR; i++) {
    const t = i / NR, r = RMAX * (Math.exp(a * t) - 1) / ea, dr = RMAX * a * Math.exp(a * t) / ea / NR;
    const sp = Math.max(dr, 2 * Math.PI * r / NS);
    for (let j = 0; j < NS; j++) { const ang = (j / NS) * Math.PI * 2; verts[k++] = r * Math.cos(ang); verts[k++] = r * Math.sin(ang); verts[k++] = sp; }
  }
  const idx = new Uint32Array(NR * NS * 6);
  k = 0;
  for (let i = 0; i < NR; i++) for (let j = 0; j < NS; j++) {
    const p = i * NS + j, q = i * NS + (j + 1) % NS, r = (i + 1) * NS + j, s = (i + 1) * NS + (j + 1) % NS;
    idx[k++] = p; idx[k++] = q; idx[k++] = s; idx[k++] = p; idx[k++] = s; idx[k++] = r;
  }
  const o = vao([3], verts);
  G.bindVertexArray(o.vao);
  const ib = G.createBuffer()!; G.bindBuffer(G.ELEMENT_ARRAY_BUFFER, ib); G.bufferData(G.ELEMENT_ARRAY_BUFFER, idx, G.STATIC_DRAW);
  G.bindVertexArray(null);
  return { vao: o.vao, count: idx.length };
})();

const land = solid(buildLand());
const shipMesh = solid(buildShipMesh());
const sailMesh = solid(buildSailMesh());
const cannonMesh = solid(buildCannonMesh());
const harpoonMesh = solid(buildHarpoonMesh());
const boltMesh = solid(buildBoltMesh());
const ballMesh = solid(buildBallMesh());
const wheelMesh = solid(buildWheelMesh());
const segMesh = solid(buildSegmentMesh());
const headMesh = solid(buildHeadMesh());
const buoyMeshes = [0, 1, 2].map((k) => solid(buildBuoy(k as 0 | 1 | 2)));
const COATS: V3[] = [[0.55, 0.08, 0.07], [0.1, 0.2, 0.5], [0.12, 0.38, 0.16], [0.4, 0.14, 0.45]];
const COAT_CSS = ['#d9473c', '#5b8de0', '#56b46a', '#b45bd0'];
const avatarMeshes = COATS.map((c) => { const p = buildAvatarParts(c); return Object.fromEntries(Object.entries(p).map(([k, v]) => [k, solid(v)])) as Record<keyof AvatarParts, Mesh>; });

const beam = (() => {
  const L = 1800, Rb = L * Math.tan(3 * Math.PI / 180), seg = 16, d: number[] = [];
  for (const sgn of [1, -1]) for (let i = 0; i < seg; i++) {
    const a0 = i / seg * Math.PI * 2, a1 = (i + 1) / seg * Math.PI * 2;
    d.push(0, 0, 0, 0, sgn * L, Math.cos(a0) * Rb, Math.sin(a0) * Rb, 1, sgn * L, Math.cos(a1) * Rb, Math.sin(a1) * Rb, 1);
  }
  return vao([4], new Float32Array(d));
})();
const glowBuf = vao([3, 3, 4], null);
const boltBuf = vao([3, 1], null);
const MAXP = isMobile ? 700 : 1400;
const sprayPos = new Float32Array(MAXP * 4), sprayVel = new Float32Array(MAXP * 3);
const sprayBuf = vao([4], sprayPos, G.DYNAMIC_DRAW);
const rainVao = G.createVertexArray()!;

// ---------- Textures ----------
const shoreData = buildShoreMap();
const shoreTex = G.createTexture()!;
G.bindTexture(G.TEXTURE_2D, shoreTex);
G.pixelStorei(G.UNPACK_ALIGNMENT, 1);
G.texImage2D(G.TEXTURE_2D, 0, G.R8, SHORE.res, SHORE.res, 0, G.RED, G.UNSIGNED_BYTE, shoreData);
G.texParameteri(G.TEXTURE_2D, G.TEXTURE_MIN_FILTER, G.LINEAR); G.texParameteri(G.TEXTURE_2D, G.TEXTURE_MAG_FILTER, G.LINEAR);
G.texParameteri(G.TEXTURE_2D, G.TEXTURE_WRAP_S, G.CLAMP_TO_EDGE); G.texParameteri(G.TEXTURE_2D, G.TEXTURE_WRAP_T, G.CLAMP_TO_EDGE);

interface RT { tex: WebGLTexture; fb: WebGLFramebuffer; }
function makeRT(size: number, internal: number, type: number): RT | null {
  const tex = G.createTexture()!;
  G.bindTexture(G.TEXTURE_2D, tex);
  G.texImage2D(G.TEXTURE_2D, 0, internal, size, size, 0, G.RGBA, type, null);
  G.texParameteri(G.TEXTURE_2D, G.TEXTURE_MIN_FILTER, G.LINEAR); G.texParameteri(G.TEXTURE_2D, G.TEXTURE_MAG_FILTER, G.LINEAR);
  G.texParameteri(G.TEXTURE_2D, G.TEXTURE_WRAP_S, G.CLAMP_TO_EDGE); G.texParameteri(G.TEXTURE_2D, G.TEXTURE_WRAP_T, G.CLAMP_TO_EDGE);
  const fb = G.createFramebuffer()!;
  G.bindFramebuffer(G.FRAMEBUFFER, fb);
  G.framebufferTexture2D(G.FRAMEBUFFER, G.COLOR_ATTACHMENT0, G.TEXTURE_2D, tex, 0);
  const ok = G.checkFramebufferStatus(G.FRAMEBUFFER) === G.FRAMEBUFFER_COMPLETE;
  G.clearColor(0, 0, 0, 0); G.clear(G.COLOR_BUFFER_BIT);
  G.bindFramebuffer(G.FRAMEBUFFER, null);
  return ok ? { tex, fb } : null;
}
const WAKE = { size: 512, res: 384, ox: 0, oy: -300 };
const mkWake = () => makeRT(WAKE.res, G.RGBA16F, G.HALF_FLOAT) ?? makeRT(WAKE.res, G.RGBA8, G.UNSIGNED_BYTE)!;
let wakeA = mkWake(), wakeB = mkWake();

// ---------- Matrix helpers ----------
function fromAxes(X: V3, Y: V3, Z: V3, p: V3): Mat4 {
  return new Float32Array([X[0], X[1], X[2], 0, Y[0], Y[1], Y[2], 0, Z[0], Z[1], Z[2], 0, p[0], p[1], p[2], 1]);
}
const SC = (x: number, y: number, z: number): Mat4 => new Float32Array([x, 0, 0, 0, 0, y, 0, 0, 0, 0, z, 0, 0, 0, 0, 1]);
const RYm = (a: number): Mat4 => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]); };
const RXm = (a: number): Mat4 => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]); };
const mul = (...ms: Mat4[]) => ms.reduce((a, b) => multiply(a, b));
const IDENT = new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
const cross = (a: V3, b: V3): V3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const sub = (a: V3, b: V3): V3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dist3 = (a: V3, b: V3) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const angLerp = (a: number, b: number, t: number) => a + Math.atan2(Math.sin(b - a), Math.cos(b - a)) * t;

// ---------- Game state ----------
const ship = new Ship();
const audio = new SeaAudio();
const net = new Net();
const monsters = new MonsterSystem();
let playing = false;
let time = 0;
let score = 0;
let spawnMode: 'shelter' | 'storm' = 'shelter';
type Mode = 'walk' | 'helm' | 'harpoon';
interface Player {
  id: number; name: string; color: number; x: number; y: number; yaw: number; pitch: number; mode: Mode;
  walk: number; phase: number; wave: number; aimYaw: number; aimPitch: number; moving: boolean;
  snaps: { t: number; x: number; y: number; yaw: number; pitch: number; ay: number; ap: number }[]; tag?: HTMLDivElement; lastStep: number;
}
const newPlayer = (id: number, name: string, color: number): Player => ({
  id, name, color, x: -6, y: 0, yaw: 0, pitch: 0, mode: 'walk', walk: 0, phase: 0, wave: 0, aimYaw: 0, aimPitch: 0.1, moving: false, snaps: [], lastStep: 0,
});
const me = newPlayer(0, 'Pirate', 0);
const others = new Map<number, Player>();
let helmId: number | null = null;
let camView: 'fp' | 'tp' = 'fp';
const isHost = () => net.role !== 'guest';
function lockPointer() {
  if (!document.hasFocus()) return;
  try {
    const r = canvas.requestPointerLock() as unknown;
    if (r instanceof Promise) r.catch(() => undefined);
  } catch { /* unsupported */ }
}

const atm = { s: 0, fog: [0, 0, 0] as V3, sky: [0, 0, 0] as V3, sun: [0, 0, 0] as V3, ambT: [0, 0, 0] as V3, ambB: [0, 0, 0] as V3, fogDen: 0, rain: 0, wind: 0 };
const windVec: [number, number] = [Math.cos(WIND_DIR), Math.sin(WIND_DIR)];
function updateAtmosphere(dt: number) {
  atm.s += (stormAt(ship.x, ship.y) - atm.s) * (1 - Math.exp(-dt / 4));
  const a = 0.15 + 0.85 * atm.s;
  atm.fog = mix3([0.3, 0.34, 0.37], [0.17, 0.2, 0.23], a);
  atm.sky = mix3([0.19, 0.24, 0.3], [0.045, 0.055, 0.07], a);
  atm.sun = mix3([0.62, 0.5, 0.34], [0.15, 0.15, 0.16], a);
  atm.ambT = mix3([0.3, 0.34, 0.38], [0.24, 0.26, 0.3], a);
  atm.ambB = [atm.ambT[0] * 0.55, atm.ambT[1] * 0.55, atm.ambT[2] * 0.6];
  atm.fogDen = mix(0.00045, 0.0017, a);
  atm.rain = smoothstep(0.05, 1, a);
  atm.wind = mix(0.2, 1, a);
}

const SPAWNS = { shelter: { x: 0, y: -300, psi: -Math.PI / 2 }, storm: { x: 1900, y: -3300, psi: 2.3 } };
function spawnShip(mode: 'shelter' | 'storm') {
  const s = SPAWNS[mode];
  updatePhases(time);
  ship.place(s.x, s.y, s.psi);
  ship.sail = mode === 'storm' ? 0.6 : 0; ship.sailSet = mode === 'storm' ? 0.6 : 0.08; ship.hull = 100;
  monsters.list = [];
  WAKE.ox = s.x; WAKE.oy = s.y;
  resetWakeMarks();
  atm.s = stormAt(s.x, s.y);
  trail.length = 0;
}

// ---------- Input ----------
const keys = new Set<string>();
const look = { dx: 0, dy: 0 };
const stick = { active: false, id: -1, x0: 0, y0: 0, x: 0, y: 0 };
let lookPtr = -1, lookX = 0, lookY = 0;
let helmRudder = 0, helmSailRate = 0;
let remoteHelm = { rudder: 0, sail: 0 };
let overview = { yaw: 0, pitch: 0 };

function interactTarget(): { kind: 'helm' | 'harpoon' | 'cannon'; idx: number; label: string } | null {
  if (me.mode !== 'walk') return null;
  if (Math.hypot(me.x - HELM.stand[0], me.y - HELM.stand[1]) < 1.6) {
    if (helmId !== null && helmId !== me.id) return { kind: 'helm', idx: -1, label: `${playerName(helmId)} has the wheel` };
    return { kind: 'helm', idx: 0, label: '<kbd>E</kbd> take the wheel' };
  }
  if (Math.hypot(me.x - (HARPOON.x - 1.9), me.y) < 1.5) {
    const busy = [...others.values()].some((o) => o.mode === 'harpoon');
    return busy ? { kind: 'harpoon', idx: -1, label: 'Harpoon is manned' } : { kind: 'harpoon', idx: 0, label: '<kbd>E</kbd> man the harpoon' };
  }
  let best = -1, bd = 1.6;
  CANNONS.forEach((c, i) => { const d = Math.hypot(me.x - c.x, me.y - (c.y - c.side * 1.5)); if (d < bd) { bd = d; best = i; } });
  if (best >= 0) {
    const ready = cannonCd[best] <= 0;
    return { kind: 'cannon', idx: best, label: ready ? `<kbd>Click</kbd> fire the ${CANNONS[best].side > 0 ? 'port' : 'starboard'} cannon` : 'Reloading…' };
  }
  return null;
}
function playerName(id: number) { return id === me.id ? 'You' : others.get(id)?.name ?? 'A mate'; }

function interact() {
  if (!playing) return;
  if (me.mode === 'helm') { leaveHelm(); return; }
  if (me.mode === 'harpoon') { me.mode = 'walk'; me.x = HARPOON.x - 1.9; me.y = 0; return; }
  const t = interactTarget();
  if (!t || t.idx < 0) return;
  if (t.kind === 'helm') {
    if (isHost()) { if (helmId === null) { helmId = me.id; enterHelm(); } }
    else net.send({ t: 'claim' });
  } else if (t.kind === 'harpoon') { me.mode = 'harpoon'; me.aimYaw = 0; me.aimPitch = 0.0; }
  else if (t.kind === 'cannon') fire('c', t.idx);
}
function enterHelm() { me.mode = 'helm'; me.x = HELM.stand[0]; me.y = HELM.stand[1]; me.yaw = 0; me.pitch = -0.05; overview = { yaw: 0, pitch: 0 }; toast('You have the wheel', 2); }
function leaveHelm() {
  me.mode = 'walk'; me.x = HELM.stand[0] - 0.8;
  if (isHost()) { if (helmId === me.id) helmId = null; } else net.send({ t: 'release' });
}
function fireAction() {
  if (!playing) return;
  if (me.mode === 'harpoon') { fire('h', 0); return; }
  const t = interactTarget();
  if (t && t.kind === 'cannon') fire('c', t.idx);
}
function wave() { me.wave = 1.6; }
function toggleCam() { camView = camView === 'fp' ? 'tp' : 'fp'; }
function toggleMute() { audio.setMuted(!audio.muted); $('muteBtn').textContent = audio.muted ? 'Sound' : 'Mute'; }

window.addEventListener('keydown', (e) => {
  if (e.target instanceof HTMLInputElement) return;
  const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
  keys.add(k);
  if (!playing || e.repeat) return;
  if (k === 'e') interact();
  if (k === 'q') wave();
  if (k === 'c') toggleCam();
  if (k === 'm') toggleMute();
  if (k === ' ' || k === 'f') { fireAction(); e.preventDefault(); }
  if (['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(k)) e.preventDefault();
});
window.addEventListener('keyup', (e) => keys.delete(e.key.length === 1 ? e.key.toLowerCase() : e.key));
window.addEventListener('blur', () => keys.clear());
canvas.addEventListener('contextmenu', (e) => e.preventDefault());
canvas.addEventListener('mousedown', (e) => {
  if (!playing || isTouch) return;
  if (document.pointerLockElement !== canvas) { lockPointer(); if (e.button !== 0) return; }
  if (e.button === 0 && document.pointerLockElement === canvas) fireAction();
});
document.addEventListener('mousemove', (e) => {
  if (document.pointerLockElement === canvas) { look.dx += e.movementX; look.dy += e.movementY; }
  else if (e.buttons & 2) { look.dx += e.movementX; look.dy += e.movementY; }
});
document.addEventListener('pointerlockchange', () => document.body.classList.toggle('locked', document.pointerLockElement === canvas));
// Touch: left half is a floating stick, right half drags the view.
canvas.addEventListener('pointerdown', (e) => {
  if (!playing || e.pointerType === 'mouse') return;
  try { canvas.setPointerCapture(e.pointerId); } catch { /* synthetic */ }
  if (e.clientX < innerWidth * 0.45 && !stick.active) {
    Object.assign(stick, { active: true, id: e.pointerId, x0: e.clientX, y0: e.clientY, x: e.clientX, y: e.clientY });
    const el = $('stick'); el.style.display = 'block'; el.style.left = `${e.clientX}px`; el.style.top = `${e.clientY}px`;
  } else if (lookPtr < 0) { lookPtr = e.pointerId; lookX = e.clientX; lookY = e.clientY; }
});
canvas.addEventListener('pointermove', (e) => {
  if (e.pointerId === stick.id && stick.active) {
    stick.x = e.clientX; stick.y = e.clientY;
    const dx = clamp(stick.x - stick.x0, -50, 50), dy = clamp(stick.y - stick.y0, -50, 50);
    $('knob').style.transform = `translate(${dx}px, ${dy}px)`;
  } else if (e.pointerId === lookPtr) { look.dx += (e.clientX - lookX) * 1.6; look.dy += (e.clientY - lookY) * 1.6; lookX = e.clientX; lookY = e.clientY; }
});
const endPtr = (e: PointerEvent) => {
  if (e.pointerId === stick.id) { stick.active = false; stick.id = -1; $('stick').style.display = 'none'; $('knob').style.transform = ''; }
  if (e.pointerId === lookPtr) lookPtr = -1;
};
canvas.addEventListener('pointerup', endPtr);
canvas.addEventListener('pointercancel', endPtr);
const tap = (id: string, fn: () => void) => $(id).addEventListener('pointerdown', (e) => { e.preventDefault(); e.stopPropagation(); fn(); });
tap('fireT', fireAction); tap('actT', interact); tap('waveT', wave);

function stickVec(): [number, number] {
  if (!stick.active) return [0, 0];
  return [clamp((stick.x - stick.x0) / 50, -1, 1), clamp(-(stick.y - stick.y0) / 50, -1, 1)];
}

function updateControls(dt: number) {
  const sens = 0.0025;
  const [sx, sy] = stickVec();
  if (me.mode === 'harpoon') {
    me.aimYaw = clamp(me.aimYaw - look.dx * sens * 0.6, -0.85, 0.85);
    me.aimPitch = clamp(me.aimPitch - look.dy * sens * 0.6, -0.2, 0.45);
  } else if (me.mode === 'helm' && camView === 'tp') {
    overview.yaw -= look.dx * sens; overview.pitch = clamp(overview.pitch - look.dy * sens, -0.5, 0.6);
  } else {
    me.yaw -= look.dx * sens; me.pitch = clamp(me.pitch - look.dy * sens, -1.3, 1.3);
  }
  look.dx = look.dy = 0;
  const kx = (keys.has('d') || keys.has('ArrowRight') ? 1 : 0) - (keys.has('a') || keys.has('ArrowLeft') ? 1 : 0);
  const ky = (keys.has('w') || keys.has('ArrowUp') ? 1 : 0) - (keys.has('s') || keys.has('ArrowDown') ? 1 : 0);
  const mx = clamp(kx + sx, -1, 1), my = clamp(ky + sy, -1, 1);
  if (me.mode === 'walk') {
    const run = keys.has('Shift') || Math.hypot(sx, sy) > 0.95 ? 5.5 : 3.2;
    const c = Math.cos(me.yaw), s = Math.sin(me.yaw);
    const fx = c * my + s * mx, fy = s * my - c * mx;
    const len = Math.hypot(fx, fy);
    // The deck tilts under you: a gentle slide downhill as she rolls and pitches.
    const slideX = -Math.sin(ship.th) * 2.2, slideY = -Math.sin(ship.ph) * 2.2;
    const vx = (len > 0.01 ? fx / Math.max(len, 1) * run : 0) + slideX, vy = (len > 0.01 ? fy / Math.max(len, 1) * run : 0) + slideY;
    [me.x, me.y] = moveOnDeck(me.x, me.y, vx * dt, vy * dt);
    me.moving = len > 0.05;
    me.walk += ((me.moving ? 1 : 0) - me.walk) * (1 - Math.exp(-dt * 8));
    if (me.moving) me.phase += dt * run * 2.4;
  } else { me.walk *= Math.exp(-dt * 8); me.moving = false; }
  if (me.mode === 'helm') {
    helmRudder += clamp(-mx - helmRudder, -2 * dt, 2 * dt);
    helmSailRate = my;
  }
  me.wave = Math.max(0, me.wave - dt);
}

// ---------- Camera ----------
const cam = { eye: [0, 0, 5] as V3, fwd: [1, 0, 0] as V3, right: [0, -1, 0] as V3, up: [0, 0, 1] as V3, fov: 1.2, near: 0.25, shake: 0 };
const camSample = newSample();
const chase = { pos: [0, -300, 30] as V3, yaw: -Math.PI / 2 };
function shipPoint(lx: number, ly: number, lz: number): V3 { return ship.local(lx, ly, lz); }
function shipDir(lx: number, ly: number, lz: number): V3 {
  const { X, Y, Z } = shipAxes(ship.psi, ship.th, ship.ph);
  return [X[0] * lx + Y[0] * ly + Z[0] * lz, X[1] * lx + Y[1] * ly + Z[1] * lz, X[2] * lx + Y[2] * ly + Z[2] * lz];
}
function setLook(eye: V3, fwd: V3, stab: number) {
  const { Z } = shipAxes(ship.psi, ship.th, ship.ph);
  const upRef = norm3([Z[0] * (1 - stab), Z[1] * (1 - stab), Z[2] * (1 - stab) + stab]);
  cam.eye = eye; cam.fwd = norm3(fwd);
  cam.right = norm3(cross(cam.fwd, upRef)); cam.up = cross(cam.right, cam.fwd);
}
function keepAboveWater(p: V3, margin: number) { const w = sampleWater(p[0], p[1], camSample).z; p[2] = Math.max(p[2], w + margin); }
function updateCamera(dt: number) {
  cam.shake *= Math.exp(-dt * 5);
  const jit = () => (Math.random() - 0.5) * cam.shake;
  const sp01 = Math.min(Math.hypot(ship.vx, ship.vy) / 8, 1);
  document.body.classList.toggle('fp', playing && camView === 'fp' && me.mode !== 'helm');
  if (!playing) {
    const a = Math.PI / 2 + 0.35 + time * 0.03, R = 70;
    cam.eye = [ship.x + Math.cos(a) * R, ship.y + Math.sin(a) * R, ship.z + 18];
    setLook(cam.eye, sub([ship.x, ship.y - 20, ship.z + 14], cam.eye), 1);
    cam.fov = 55 * Math.PI / 180; cam.near = 1;
    return;
  }
  if (me.mode === 'harpoon') {
    const dl: V3 = [Math.cos(me.aimYaw) * Math.cos(me.aimPitch), Math.sin(me.aimYaw) * Math.cos(me.aimPitch), Math.sin(me.aimPitch)];
    const eye = shipPoint(HARPOON.x - dl[0] * 3.2, -dl[1] * 3.2, HARPOON.z + 1.1 - dl[2] * 3.2);
    setLook([eye[0] + jit(), eye[1] + jit(), eye[2] + jit()], shipDir(dl[0], dl[1], dl[2] - 0.05), 0.3);
    cam.fov = 62 * Math.PI / 180; cam.near = 0.3;
    return;
  }
  if (me.mode === 'helm' && camView === 'tp') {
    chase.yaw = angLerp(chase.yaw, ship.psi, 1 - Math.exp(-dt * 1.2));
    const yaw = chase.yaw + overview.yaw, d = 75, h = 30 - overview.pitch * 40;
    const want: V3 = [ship.x - Math.cos(yaw) * d, ship.y - Math.sin(yaw) * d, ship.z + h];
    const k = 1 - Math.exp(-dt * 2.5);
    chase.pos = [chase.pos[0] + (want[0] - chase.pos[0]) * k, chase.pos[1] + (want[1] - chase.pos[1]) * k, chase.pos[2] + (want[2] - chase.pos[2]) * k];
    keepAboveWater(chase.pos, 4);
    cam.eye = [chase.pos[0] + jit(), chase.pos[1] + jit(), chase.pos[2] + jit()];
    setLook(cam.eye, sub([ship.x + Math.cos(ship.psi) * 10, ship.y + Math.sin(ship.psi) * 10, ship.z + 10], cam.eye), 1);
    cam.fov = (58 + 6 * sp01) * Math.PI / 180; cam.near = 1;
    return;
  }
  const z0 = deckZ(me.x, me.y);
  const dl: V3 = [Math.cos(me.yaw) * Math.cos(me.pitch), Math.sin(me.yaw) * Math.cos(me.pitch), Math.sin(me.pitch)];
  if (camView === 'fp') {
    const bob = Math.abs(Math.sin(me.phase)) * 0.06 * me.walk;
    const eye = shipPoint(me.x, me.y, z0 + 1.68 + bob);
    setLook([eye[0] + jit(), eye[1] + jit(), eye[2] + jit()], shipDir(dl[0], dl[1], dl[2]), 0.35);
    cam.fov = (70 + 4 * sp01) * Math.PI / 180; cam.near = 0.2;
  } else {
    const back: V3 = [me.x - Math.cos(me.yaw) * 5.5, me.y - Math.sin(me.yaw) * 5.5, z0 + 2.6 - Math.sin(me.pitch) * 3];
    const eye = shipPoint(back[0], back[1], back[2]);
    keepAboveWater(eye, 1.5);
    const tgt = shipPoint(me.x, me.y, z0 + 1.6);
    setLook([eye[0] + jit(), eye[1] + jit(), eye[2] + jit()], sub(tgt, eye), 0.35);
    cam.fov = 66 * Math.PI / 180; cam.near = 0.3;
  }
}

// ---------- Spray ----------
let sprayNext = 0;
function emit(p: V3, v: V3, life = 1) {
  const i = sprayNext; sprayNext = (sprayNext + 1) % MAXP;
  sprayPos[i * 4] = p[0]; sprayPos[i * 4 + 1] = p[1]; sprayPos[i * 4 + 2] = p[2]; sprayPos[i * 4 + 3] = life;
  sprayVel[i * 3] = v[0]; sprayVel[i * 3 + 1] = v[1]; sprayVel[i * 3 + 2] = v[2];
}
function emitLocal(l: V3, lv: V3, life = 1) {
  const p = shipPoint(l[0], l[1], l[2]), v = shipDir(lv[0], lv[1], lv[2]);
  emit(p, [ship.vx + v[0], ship.vy + v[1], ship.vz + v[2]], life);
}
const R = () => Math.random();
function burst(p: V3, n: number, speed: number, up: number, life = 1) {
  for (let k = 0; k < n; k++) emit([p[0] + (R() - 0.5) * 2, p[1] + (R() - 0.5) * 2, p[2]], [(R() - 0.5) * speed, (R() - 0.5) * speed, up * (0.4 + R())], life);
}
function updateSpray(dt: number) {
  const drag = Math.exp(-dt * 0.7);
  for (let i = 0; i < MAXP; i++) {
    if (sprayPos[i * 4 + 3] <= 0) continue;
    sprayVel[i * 3 + 2] -= 9.81 * dt;
    sprayVel[i * 3] *= drag; sprayVel[i * 3 + 1] *= drag;
    sprayPos[i * 4] += sprayVel[i * 3] * dt; sprayPos[i * 4 + 1] += sprayVel[i * 3 + 1] * dt;
    const z = sprayPos[i * 4 + 2] += sprayVel[i * 3 + 2] * dt;
    let life = sprayPos[i * 4 + 3] - dt / 1.9;
    if (z < ship.z - 30) life = 0;
    sprayPos[i * 4 + 3] = life;
  }
}

// ---------- Weapons ----------
interface Shot { id: number; kind: 'c' | 'h'; p: V3; v: V3; life: number; skips: number; }
const shots: Shot[] = [];
let shotId = 1;
const cannonCd = CANNONS.map(() => 0);
const recoil = CANNONS.map(() => 0);
let harpoonCd = 0;
const shotSample = newSample();
// Aim assist: if a serpent sits inside the gun's cone, solve elevation and lead so the shot meets it.
function assist(p: V3, dir: V3, speed: number, cone: number, range: number): V3 | null {
  let best: { t: V3; score: number } | null = null;
  for (const m of monsters.list) {
    if (m.st === MS.Dying || !m.segs.length) continue;
    for (let i = 0; i < m.segs.length; i += 2) {
      const q = m.segs[i];
      const dx = q[0] - p[0], dy = q[1] - p[1], dh = Math.hypot(dx, dy);
      if (dh > range || dh < 5) continue;
      const ang = Math.acos(clamp((dx * dir[0] + dy * dir[1]) / (dh * Math.hypot(dir[0], dir[1])), -1, 1));
      if (ang > cone) continue;
      const score = ang * 40 + dh * 0.1 + i;
      if (!best || score < best.score) {
        const tof = dh / speed;
        best = { t: [q[0] + Math.cos(m.hd) * m.sp * tof * (i === 0 ? 1 : 0.6), q[1] + Math.sin(m.hd) * m.sp * tof * (i === 0 ? 1 : 0.6), q[2]], score };
      }
    }
  }
  if (!best) return null;
  const dx = best.t[0] - p[0], dy = best.t[1] - p[1], x = Math.hypot(dx, dy), y = best.t[2] - p[2], g = 9.81, v2 = speed * speed;
  const disc = v2 * v2 - g * (g * x * x + 2 * y * v2);
  if (disc < 0) return null;
  const th = clamp(Math.atan((v2 - Math.sqrt(disc)) / (g * x)), -0.35, 0.35);
  return [dx / x * Math.cos(th), dy / x * Math.cos(th), Math.sin(th)];
}
function muzzle(kind: 'c' | 'h', idx: number, aimYaw: number, aimPitch: number): { p: V3; v: V3 } {
  if (kind === 'c') {
    const c = CANNONS[idx], elev = 0.0;
    const p = shipPoint(c.x, c.y + c.side * 1.8, c.z + 0.75);
    // Gun crews time the roll: the shot follows the ship's heading but only part of its roll.
    const raw = shipDir(0, c.side * Math.cos(elev), Math.sin(elev));
    const lvl: V3 = [-Math.sin(ship.psi) * c.side * Math.cos(elev), Math.cos(ship.psi) * c.side * Math.cos(elev), Math.sin(elev)];
    const d = assist(p, lvl, 72, 0.5, 140) ?? norm3([raw[0] * 0.35 + lvl[0] * 0.65, raw[1] * 0.35 + lvl[1] * 0.65, raw[2] * 0.35 + lvl[2] * 0.65]);
    return { p, v: [d[0] * 72 + ship.vx, d[1] * 72 + ship.vy, d[2] * 72 + ship.vz] };
  }
  const dl: V3 = [Math.cos(aimYaw) * Math.cos(aimPitch), Math.sin(aimYaw) * Math.cos(aimPitch), Math.sin(aimPitch)];
  const p = shipPoint(HARPOON.x + dl[0] * 2.2, dl[1] * 2.2, HARPOON.z + 0.2 + dl[2] * 2.2);
  const raw = shipDir(dl[0], dl[1], dl[2]);
  const d = assist(p, raw, 80, 0.18, 220) ?? raw;
  return { p, v: [d[0] * 80 + ship.vx, d[1] * 80 + ship.vy, d[2] * 80 + ship.vz] };
}
function fire(kind: 'c' | 'h', idx: number) {
  if (isHost()) hostFire(kind, idx, me.aimYaw, me.aimPitch);
  else net.send({ t: 'fire', k: kind, i: idx, ay: me.aimYaw, ap: me.aimPitch });
}
function hostFire(kind: 'c' | 'h', idx: number, ay: number, ap: number) {
  if (kind === 'c' ? cannonCd[idx] > 0 : harpoonCd > 0) return;
  if (kind === 'c') cannonCd[idx] = 3; else harpoonCd = 3.5;
  const { p, v } = muzzle(kind, idx, ay, ap);
  const id = shotId++;
  spawnShot(id, kind, idx, p, v);
  net.send({ t: 'shot', id, k: kind, i: idx, p: p.map((x) => +x.toFixed(2)), v: v.map((x) => +x.toFixed(2)) });
}
function spawnShot(id: number, kind: 'c' | 'h', idx: number, p: V3, v: V3) {
  shots.push({ id, kind, p: [...p] as V3, v: [...v] as V3, life: 8, skips: 0 });
  const d = dist3(p, cam.eye);
  if (kind === 'c') {
    recoil[idx] = 1; cannonCd[idx] = Math.max(cannonCd[idx], 3);
    audio.cannon(d);
    cam.shake = Math.max(cam.shake, 0.12 / (1 + d / 15));
    const dir = norm3(v);
    for (let k = 0; k < 40; k++) emit([p[0] + dir[0] * 0.5, p[1] + dir[1] * 0.5, p[2]], [dir[0] * (8 + R() * 12) + (R() - 0.5) * 4, dir[1] * (8 + R() * 12) + (R() - 0.5) * 4, (R() - 0.2) * 4], 0.5);
  } else { harpoonCd = Math.max(harpoonCd, 3.5); audio.harpoon(d); }
}
function updateShots(dt: number) {
  for (const s of shots) {
    s.v[2] -= 9.81 * dt;
    s.p[0] += s.v[0] * dt; s.p[1] += s.v[1] * dt; s.p[2] += s.v[2] * dt;
    s.life -= dt;
    const w = sampleWater(s.p[0], s.p[1], shotSample).z;
    if (isHost()) {
      // Direct hits, or a splash close enough beside the serpent, both count.
      const m = monsters.hitTest(s.p, s.kind === 'h' ? 2.0 : 2.4) ?? (s.p[2] < w ? monsters.hitTest([s.p[0], s.p[1], w], 4.5) : null);
      if (m) {
        const killed = monsters.damage(m, s.kind === 'h' ? 2 : 1);
        s.life = 0;
        onHit(s.id, s.p, m.id, killed);
        net.send({ t: 'hit', id: s.id, p: s.p.map((x) => +x.toFixed(1)), m: m.id, k: killed ? 1 : 0 });
        if (killed) { score++; }
        continue;
      }
    }
    if (s.p[2] < w) {
      // Fast, flat cannonballs skip off the water like stones.
      const vh = Math.hypot(s.v[0], s.v[1]);
      if (s.kind === 'c' && s.skips < 3 && vh > 30 && Math.abs(s.v[2]) < vh * 0.3) {
        s.skips++; s.p[2] = w + 0.05; s.v[2] = Math.abs(s.v[2]) * 0.45 + 2.5; s.v[0] *= 0.8; s.v[1] *= 0.8;
        burst([s.p[0], s.p[1], w], 18, 4, 7, 0.8);
        audio.splash(0.35, dist3(s.p, cam.eye));
        continue;
      }
      s.life = 0;
      burst([s.p[0], s.p[1], w], s.kind === 'c' ? 50 : 25, 5, 12, 1);
      audio.splash(s.kind === 'c' ? 0.9 : 0.5, dist3(s.p, cam.eye));
    }
  }
  for (let i = shots.length - 1; i >= 0; i--) if (shots[i].life <= 0) shots.splice(i, 1);
  for (let i = 0; i < CANNONS.length; i++) { cannonCd[i] = Math.max(0, cannonCd[i] - dt); recoil[i] = Math.max(0, recoil[i] - dt * 1.5); }
  harpoonCd = Math.max(0, harpoonCd - dt);
}
function onHit(id: number, p: V3, mid: number, killed: boolean) {
  const s = shots.find((x) => x.id === id); if (s) s.life = 0;
  const d = dist3(p, cam.eye);
  audio.hit(d);
  audio.roar(d, true);
  burst(p, 60, 10, 10, 1.2);
  if (killed) { toast('Serpent slain!', 2.5); const m = monsters.list.find((x) => x.id === mid); if (m) burst(m.segs[0] ?? p, 120, 14, 16, 1.5); }
}
function onRam(m: Monster, side: number) {
  const { Y } = shipAxes(ship.psi, ship.th, ship.ph);
  ship.impulse(-Y[0] * side * 1.6, -Y[1] * side * 1.6, -side * 0.18);
  ship.hull = Math.max(0, ship.hull - 12);
  rammed(m.segs[0] ?? [m.x, m.y, ship.z]);
  net.send({ t: 'ram', p: (m.segs[0] ?? [m.x, m.y, 0]).map((x) => +x.toFixed(1)) });
}
function rammed(p: V3) {
  audio.crunch(1); audio.roar(dist3(p, cam.eye));
  cam.shake = Math.max(cam.shake, 0.35);
  burst(p, 100, 12, 14, 1.4);
  toast('Rammed!', 1.5);
}

// ---------- Lightning ----------
const bolt = { t: 99, I: 0, pts: [] as V3[][], mid: [0, 0, 500] as V3 };
let nextStrike = 5, flash = 0;
const flashDir: V3 = [0, 0, 1];
function jagged(a: V3, b: V3, levels: number, rough: number): V3[] {
  let pts: V3[] = [a, b];
  for (let l = 0; l < levels; l++) {
    const np: V3[] = [pts[0]];
    for (let k = 0; k < pts.length - 1; k++) {
      const p = pts[k], q = pts[k + 1], len = dist3(p, q);
      np.push([(p[0] + q[0]) / 2 + (R() - 0.5) * len * rough, (p[1] + q[1]) / 2 + (R() - 0.5) * len * rough, (p[2] + q[2]) / 2 + (R() - 0.5) * len * 0.15], q);
    }
    pts = np;
  }
  return pts;
}
function strike() {
  const away = Math.atan2(ship.y - IC.y, ship.x - IC.x) + (R() - 0.5) * 2.6;
  const d = mix(2600, 900, atm.s) + R() * mix(4200, 3000, atm.s);
  const x = cam.eye[0] + Math.cos(away) * d, y = cam.eye[1] + Math.sin(away) * d;
  const main = jagged([x, y, 1150], [x + (R() - 0.5) * 300, y + (R() - 0.5) * 300, 0], 7, 0.55);
  bolt.pts = [main];
  for (let b = 0; b < 3; b++) {
    const s = main[Math.floor(R() * main.length * 0.6)];
    bolt.pts.push(jagged(s, [s[0] + (R() - 0.5) * 500, s[1] + (R() - 0.5) * 500, s[2] - 200 - R() * 300], 5, 0.5));
  }
  bolt.mid = [x, y, 550]; bolt.t = 0;
  bolt.I = clamp(2.4 / (1 + d / 1500), 0.35, 1.7);
  audio.thunder(d);
}
function updateLightning(dt: number) {
  nextStrike -= dt;
  if (nextStrike <= 0) { strike(); nextStrike = mix(24, 3.5, atm.s) * (0.45 + R()); }
  bolt.t += dt;
  const t = bolt.t;
  flash = bolt.I * (t < 0.05 ? 1 : t < 0.11 ? 0.25 : t < 0.19 ? 0.85 : t < 0.25 ? 0.2 : 0.55 * Math.exp(-(t - 0.25) * 7));
  const d = norm3(sub(bolt.mid, cam.eye)); flashDir[0] = d[0]; flashDir[1] = d[1]; flashDir[2] = d[2];
}

// ---------- Wake ----------
const prevMarks = { stern: [0, 0] as number[], port: [0, 0] as number[], stbd: [0, 0] as number[] };
let wakeBlob = 0;
const segU = new Float32Array(12), segP = new Float32Array(6);
function wakePass(p: Program) {
  G.bindFramebuffer(G.FRAMEBUFFER, wakeB.fb); G.viewport(0, 0, WAKE.res, WAKE.res);
  G.activeTexture(G.TEXTURE0); G.bindTexture(G.TEXTURE_2D, wakeA.tex);
  p.i('uSrc', 0);
  G.bindVertexArray(quad.vao); G.drawArrays(G.TRIANGLE_STRIP, 0, 4);
  [wakeA, wakeB] = [wakeB, wakeA];
}
function updateWake() {
  const dx = Math.round(ship.x - WAKE.ox), dy = Math.round(ship.y - WAKE.oy);
  if (Math.abs(dx) > 400 || Math.abs(dy) > 400) { WAKE.ox = Math.round(ship.x); WAKE.oy = Math.round(ship.y); resetWakeMarks(); }
  else if (Math.abs(dx) > 90 || Math.abs(dy) > 90) {
    wakePass(pShift.use().v2('uShift', dx / WAKE.size, dy / WAKE.size));
    WAKE.ox += dx; WAKE.oy += dy;
  }
  const uv = (p: number[]) => [(p[0] - WAKE.ox) / WAKE.size + 0.5, (p[1] - WAKE.oy) / WAKE.size + 0.5];
  const st = ship.local(-18.5, 0, 0), po = ship.local(13, 3.6, 0), sb = ship.local(13, -3.6, 0);
  const speed = Math.hypot(ship.vx, ship.vy);
  const marks: [number[], V3, number, number][] = [
    [prevMarks.stern, st, 3.5 + speed * 0.2, 0.012 * clamp(speed / 6, 0, 1.3)],
    [prevMarks.port, po, 1.6, 0.012 * smoothstep(1.5, 7, speed)],
    [prevMarks.stbd, sb, 1.6, 0.012 * smoothstep(1.5, 7, speed)],
  ];
  marks.forEach(([prev, now, w, I], i) => {
    const a = uv(prev), b = uv(now);
    segU.set([a[0], a[1], b[0], b[1]], i * 4); segP.set([w / WAKE.size, I], i * 2);
    prev[0] = now[0]; prev[1] = now[1];
  });
  const bow = uv(ship.local(17, 0, 0));
  wakePass(pWake.use().v2('uTexel', 1 / WAKE.res, 1 / WAKE.res).v4a('uSeg', segU).v2a('uSegP', segP)
    .v3('uBlob', [bow[0], bow[1], 9 / WAKE.size]).f('uBlobI', wakeBlob));
  wakeBlob = 0;
}
function resetWakeMarks() {
  const st = ship.local(-18.5, 0, 0), po = ship.local(13, 3.6, 0), sb = ship.local(13, -3.6, 0);
  prevMarks.stern = [st[0], st[1]]; prevMarks.port = [po[0], po[1]]; prevMarks.stbd = [sb[0], sb[1]];
}

// ---------- Buoys ----------
const buoySample = newSample(), buoyGx = newSample(), buoyGy = newSample();
const buoyModels: Mat4[] = BUOYS.map(() => IDENT);
const buoyTops: V3[] = BUOYS.map(() => [0, 0, 0]);
function updateBuoys() {
  BUOYS.forEach((b, i) => {
    if (Math.hypot(b.x - cam.eye[0], b.y - cam.eye[1]) > 2500) return;
    const z = sampleWater(b.x, b.y, buoySample).z;
    const gx = (sampleWater(b.x + 1.5, b.y, buoyGx).z - z) / 1.5, gy = (sampleWater(b.x, b.y + 1.5, buoyGy).z - z) / 1.5;
    const ax = shipAxes(b.phase, Math.atan(gx * 0.8), -Math.atan(gy * 0.8));
    const p: V3 = [b.x, b.y, z - 0.2];
    buoyModels[i] = fromAxes(ax.X, ax.Y, ax.Z, p);
    buoyTops[i] = [p[0] + ax.Z[0] * 3.7, p[1] + ax.Z[1] * 3.7, p[2] + ax.Z[2] * 3.7];
  });
}

// ---------- Glows ----------
const MAXG = 64;
const glowData = new Float32Array(MAXG * 60);
let glowCount = 0;
const CORNERS = [[-1, -1], [1, -1], [1, 1], [-1, -1], [1, 1], [-1, 1]];
function addGlow(p: V3, size: number, col: V3, a: number) {
  if (a <= 0.002 || glowCount >= MAXG) return;
  CORNERS.forEach((c, k) => glowData.set([p[0], p[1], p[2], c[0], c[1], size, col[0], col[1], col[2], a], glowCount * 60 + k * 10));
  glowCount++;
}
const BUOY_LIGHT: V3[] = [[1, 0.25, 0.2], [0.3, 1, 0.45], [1, 0.8, 0.3]];

// ---------- HUD ----------
const compass = $<HTMLCanvasElement>('compass'), cctx = compass.getContext('2d')!;
const minimap = $<HTMLCanvasElement>('minimap'), mctx = minimap.getContext('2d')!;
const islandImg = (() => {
  const res = SHORE.res, out = document.createElement('canvas');
  out.width = res; out.height = res;
  const ctx = out.getContext('2d')!, img = ctx.createImageData(res, res);
  for (let j = 0; j < res; j++) for (let i = 0; i < res; i++) {
    const h = shoreData[j * res + i] / 255 * 40 - 20, o = ((res - 1 - j) * res + i) * 4;
    if (h > 0.3) img.data.set([118, 126, 112, 255], o);
    else if (h > -8) img.data.set([60, 98, 104, Math.round(200 * (1 + h / 8))], o);
  }
  ctx.putImageData(img, 0, 0);
  return out;
})();
const trail: number[][] = [];
let trailT = 0;
const knotsEl = $('knots'), seaEl = $('sea'), leverFill = $('leverFill'), needle = $('rudderNeedle'), toastEl = $('toast');
const promptEl = $('prompt'), hullFill = $('hullfill'), killsEl = $('kills'), crewEl = $('crew'), tagsEl = $('tags');
let toastTimer = 0;
function toast(msg: string, dur = 3.5) { toastEl.textContent = msg; toastEl.classList.add('show'); toastTimer = dur; }
const bearing = (dx: number, dy: number) => ((Math.atan2(dx, dy) * 180 / Math.PI) + 360) % 360;
const SEA_STATES = ['Sheltered water', 'Moderate swell', 'Rough sea', 'Very rough', 'High seas'];
let seaState = -1, seaCand = 0, seaCandT = 0;

function drawCompass(hdg: number) {
  const W = compass.width, Hh = compass.height;
  if (!W) return;
  const dpr = W / compass.clientWidth, ppd = W / 120;
  cctx.clearRect(0, 0, W, Hh);
  const g = cctx.createLinearGradient(0, 0, W, 0);
  g.addColorStop(0, 'rgba(8,14,20,0)'); g.addColorStop(0.15, 'rgba(8,14,20,0.45)'); g.addColorStop(0.85, 'rgba(8,14,20,0.45)'); g.addColorStop(1, 'rgba(8,14,20,0)');
  cctx.fillStyle = g; cctx.fillRect(0, 0, W, Hh);
  cctx.textAlign = 'center'; cctx.textBaseline = 'top';
  const names: Record<number, string> = { 0: 'N', 45: 'NE', 90: 'E', 135: 'SE', 180: 'S', 225: 'SW', 270: 'W', 315: 'NW' };
  for (let d = Math.ceil((hdg - 60) / 5) * 5; d <= hdg + 60; d += 5) {
    const x = W / 2 + (d - hdg) * ppd, dd = ((d % 360) + 360) % 360;
    cctx.globalAlpha = Math.max(1 - Math.abs(d - hdg) / 62, 0);
    cctx.fillStyle = '#e4ebf0';
    const major = dd % 15 === 0;
    cctx.fillRect(x - dpr * 0.5, Hh - (major ? 9 : 5) * dpr, dpr, (major ? 9 : 5) * dpr);
    if (names[dd] !== undefined) { cctx.font = `600 ${12 * dpr}px system-ui`; cctx.fillText(names[dd], x, 3 * dpr); }
    else if (major) { cctx.font = `${9 * dpr}px system-ui`; cctx.fillStyle = 'rgba(228,235,240,0.65)'; cctx.fillText(String(dd), x, 5 * dpr); }
  }
  cctx.globalAlpha = 1;
  const lb = bearing(LIGHTHOUSE.x - ship.x, LIGHTHOUSE.y - ship.y);
  let rel = ((lb - hdg + 540) % 360) - 180;
  const edge = Math.abs(rel) > 58;
  rel = clamp(rel, -58, 58);
  const mx = W / 2 + rel * ppd;
  cctx.fillStyle = '#e9b44c';
  cctx.beginPath();
  if (edge) { const s = Math.sign(rel); cctx.moveTo(mx + s * 6 * dpr, Hh / 2); cctx.lineTo(mx - s * 2 * dpr, Hh / 2 - 6 * dpr); cctx.lineTo(mx - s * 2 * dpr, Hh / 2 + 6 * dpr); }
  else { cctx.moveTo(mx, Hh - 11 * dpr); cctx.lineTo(mx - 5 * dpr, Hh - 19 * dpr); cctx.lineTo(mx + 5 * dpr, Hh - 19 * dpr); }
  cctx.fill();
  cctx.fillStyle = '#e4ebf0';
  cctx.fillRect(W / 2 - dpr, Hh - 14 * dpr, 2 * dpr, 14 * dpr);
}

function drawMinimap() {
  const W = minimap.width;
  if (!W) return;
  const dpr = W / minimap.clientWidth, cx = W / 2, cy = W / 2, rad = W / 2 - 2 * dpr;
  const scale = rad / 1300;
  const mx = (x: number) => cx + (x - ship.x) * scale, my = (y: number) => cy - (y - ship.y) * scale;
  mctx.clearRect(0, 0, W, W);
  mctx.save();
  mctx.beginPath(); mctx.arc(cx, cy, rad, 0, Math.PI * 2); mctx.clip();
  const g = mctx.createRadialGradient(mx(IC.x), my(IC.y), 600 * scale, mx(IC.x), my(IC.y), 3000 * scale);
  g.addColorStop(0, 'rgba(34,62,72,0.88)'); g.addColorStop(1, 'rgba(8,13,18,0.92)');
  mctx.fillStyle = g; mctx.fillRect(0, 0, W, W);
  mctx.setLineDash([4 * dpr, 4 * dpr]); mctx.strokeStyle = 'rgba(233,180,76,0.45)'; mctx.lineWidth = dpr;
  mctx.beginPath(); mctx.arc(mx(IC.x), my(IC.y), 1500 * scale, 0, Math.PI * 2); mctx.stroke(); mctx.setLineDash([]);
  const sz = SHORE.size * scale;
  mctx.drawImage(islandImg, mx(IC.x + SHORE.min), my(IC.y + SHORE.min + SHORE.size), sz, sz);
  mctx.fillStyle = '#9aa196';
  for (const p of PILLARS) { mctx.beginPath(); mctx.arc(mx(p.x), my(p.y), Math.max(p.r * scale, 1.5 * dpr), 0, Math.PI * 2); mctx.fill(); }
  BUOYS.forEach((b) => { const c = BUOY_COLS[b.kind]; mctx.fillStyle = `rgb(${c.map((v) => Math.min(255, Math.round(v * 320))).join(',')})`; mctx.fillRect(mx(b.x) - dpr, my(b.y) - dpr, 2 * dpr, 2 * dpr); });
  mctx.fillStyle = `rgba(255,214,120,${Math.sin(time * 1.1) > 0.6 ? 1 : 0.55})`; mctx.beginPath(); mctx.arc(mx(LIGHTHOUSE.x), my(LIGHTHOUSE.y), 3 * dpr, 0, Math.PI * 2); mctx.fill();
  mctx.fillStyle = '#e0624a';
  for (const m of monsters.list) if (m.st !== MS.Dying) { mctx.beginPath(); mctx.arc(mx(m.x), my(m.y), 3 * dpr, 0, Math.PI * 2); mctx.fill(); }
  mctx.lineWidth = 1.5 * dpr;
  for (let i = 1; i < trail.length; i++) {
    mctx.strokeStyle = `rgba(200,225,235,${(i / trail.length) * 0.6})`;
    mctx.beginPath(); mctx.moveTo(mx(trail[i - 1][0]), my(trail[i - 1][1])); mctx.lineTo(mx(trail[i][0]), my(trail[i][1])); mctx.stroke();
  }
  mctx.restore();
  mctx.strokeStyle = 'rgba(228,235,240,0.35)'; mctx.lineWidth = dpr;
  mctx.beginPath(); mctx.arc(cx, cy, rad, 0, Math.PI * 2); mctx.stroke();
  mctx.fillStyle = '#e4ebf0'; mctx.font = `600 ${10 * dpr}px system-ui`; mctx.textAlign = 'center'; mctx.fillText('N', cx, 13 * dpr);
  mctx.save(); mctx.translate(cx, cy); mctx.rotate(-ship.psi);
  mctx.fillStyle = '#e9b44c'; mctx.beginPath(); mctx.moveTo(8 * dpr, 0); mctx.lineTo(-5 * dpr, 4.5 * dpr); mctx.lineTo(-3 * dpr, 0); mctx.lineTo(-5 * dpr, -4.5 * dpr); mctx.fill();
  mctx.restore();
}

let gatePrev = [ship.x, ship.y];
function checkGates() {
  for (let i = 0; i < PILLARS.length - 1; i++) {
    const A = PILLARS[i], B = PILLARS[i + 1];
    const ex = B.x - A.x, ey = B.y - A.y;
    const s0 = ex * (gatePrev[1] - A.y) - ey * (gatePrev[0] - A.x), s1 = ex * (ship.y - A.y) - ey * (ship.x - A.x);
    if (s0 * s1 >= 0) continue;
    const t = ((ship.x - A.x) * ex + (ship.y - A.y) * ey) / (ex * ex + ey * ey);
    if (t < 0 || t > 1) continue;
    const islandSide = ex * (IC.y - A.y) - ey * (IC.x - A.x);
    const outward = (s1 > 0) !== (islandSide > 0);
    toast(i === ARCH.a ? (outward ? 'Through the arch' : 'Home through the arch') : outward ? 'Past the stacks' : 'Back inside the bay');
  }
  gatePrev = [ship.x, ship.y];
}

function updateCrewPanel() {
  const rows: string[] = [];
  const role = (id: number) => (helmId === id ? '· at the wheel' : '');
  rows.push(`<div class="who"><span class="dot" style="background:${COAT_CSS[me.color]}"></span>${esc(me.name)} <span class="role">(you) ${role(me.id)}</span></div>`);
  for (const o of others.values()) rows.push(`<div class="who"><span class="dot" style="background:${COAT_CSS[o.color]}"></span>${esc(o.name)} <span class="role">${role(o.id)}</span></div>`);
  if (net.online) rows.push(`<button id="invite">Copy invite link · ${net.room}</button>`);
  crewEl.innerHTML = rows.join('');
  const inv = document.getElementById('invite');
  if (inv) inv.onclick = () => { void navigator.clipboard?.writeText(inviteLink()).then(() => toast('Invite link copied', 2)); };
}
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => `&#${c.charCodeAt(0)};`);
const inviteLink = () => `${location.origin}${location.pathname}?crew=${net.room}`;

function updateHud(dt: number) {
  const sp = Math.hypot(ship.vx, ship.vy);
  knotsEl.textContent = (sp * 1.944).toFixed(0);
  const hs = sigWaveHeight(ship.x, ship.y);
  const s = stormAt(ship.x, ship.y);
  const cat = s < 0.06 ? 0 : s < 0.28 ? 1 : s < 0.52 ? 2 : s < 0.78 ? 3 : 4;
  if (cat !== seaCand) { seaCand = cat; seaCandT = 0; } else seaCandT += dt;
  if (seaCand !== seaState && (seaCandT > 1.5 || seaState < 0)) { if (seaState >= 0 && playing) toast(SEA_STATES[seaCand]); seaState = seaCand; }
  seaEl.innerHTML = `${SEA_STATES[Math.max(seaState, 0)]} · waves <b>${hs.toFixed(1)} m</b>`;
  leverFill.style.height = `${clamp((ship.sail), 0, 1) * 100}%`;
  needle.style.left = `${50 - ship.rudder * 45}%`;
  hullFill.style.width = `${ship.hull}%`;
  hullFill.style.background = ship.hull > 60 ? '#7fc7a0' : ship.hull > 30 ? '#e9b44c' : '#e0624a';
  killsEl.textContent = `Hull ${Math.round(ship.hull)}% · serpents slain ${score}`;
  drawCompass(((90 - ship.psi * 180 / Math.PI) % 360 + 360) % 360);
  trailT += dt;
  if (trailT > 0.5) { trailT = 0; trail.push([ship.x, ship.y]); if (trail.length > 160) trail.shift(); }
  drawMinimap();
  // Context prompt and touch buttons.
  const t = interactTarget();
  let msg = t?.label ?? '';
  if (me.mode === 'helm') msg = isTouch ? 'Stick: steer and set sail · Use: leave the wheel' : '<kbd>W</kbd>/<kbd>S</kbd> sails · <kbd>A</kbd>/<kbd>D</kbd> helm · <kbd>C</kbd> overview · <kbd>E</kbd> leave';
  if (me.mode === 'harpoon') msg = harpoonCd > 0 ? 'Reloading…' : isTouch ? 'Drag to aim · Fire' : 'Mouse to aim · <kbd>Click</kbd> fire · <kbd>E</kbd> leave';
  if (isTouch) msg = msg.replace(/<kbd>Click<\/kbd> /, '').replace(/<kbd>E<\/kbd> /, '');
  promptEl.innerHTML = msg; promptEl.classList.toggle('show', !!msg && playing);
  $('fireT').style.display = (t?.kind === 'cannon' || me.mode === 'harpoon') ? 'block' : 'none';
  $('actT').style.display = ((t && t.idx >= 0 && t.kind !== 'cannon') || me.mode !== 'walk') ? 'block' : 'none';
  $('actT').textContent = me.mode === 'walk' ? (t?.kind === 'helm' ? 'Wheel' : 'Harpoon') : 'Leave';
  if (toastTimer > 0) { toastTimer -= dt; if (toastTimer <= 0) toastEl.classList.remove('show'); }
}

// ---------- Resize ----------
function resize() {
  const dpr = Math.min(devicePixelRatio || 1, isMobile ? 1.25 : 1.5);
  const w = Math.floor(innerWidth * dpr), h = Math.floor(innerHeight * dpr);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  const hd = Math.min(devicePixelRatio || 1, 2);
  for (const c of [compass, minimap]) {
    const cw = Math.round(c.clientWidth * hd), ch = Math.round(c.clientHeight * hd);
    if (cw && (c.width !== cw || c.height !== ch)) { c.width = cw; c.height = ch; }
  }
}
function fitFov(vfov: number, aspect: number) {
  const hmin = 78 * Math.PI / 180;
  return Math.min(Math.max(vfov, 2 * Math.atan(Math.tan(hmin / 2) / aspect)), 105 * Math.PI / 180);
}

// ---------- Render ----------
function common(p: Program) {
  return p.v3('uCamPos', cam.eye).v3('uSunDir', SUN_DIR).v3('uSunCol', atm.sun).v3('uAmbTop', atm.ambT).v3('uAmbBot', atm.ambB)
    .v3('uFogCol', atm.fog).f('uFogDen', atm.fogDen).f('uFlash', flash).v3('uFlashDir', flashDir).f('uTime', time);
}
let curVP: Mat4 = IDENT;
function draw(mesh: Mesh, model: Mat4, wet = 0) {
  pSolid.m4('uModel', model).f('uWet', wet);
  G.bindVertexArray(mesh.vao); G.drawArrays(G.TRIANGLES, 0, mesh.count);
}
function drawAvatar(p: Player, shipModel: Mat4) {
  const meshes = avatarMeshes[p.color % avatarMeshes.length];
  const hold = p.mode === 'walk' ? 0 : 1;
  const yaw = p.mode === 'helm' ? 0 : p.mode === 'harpoon' ? p.aimYaw : p.yaw;
  const px = p.mode === 'harpoon' ? HARPOON.x - Math.cos(p.aimYaw) * 1.9 : p.x, py = p.mode === 'harpoon' ? -Math.sin(p.aimYaw) * 1.9 : p.y;
  const base = mul(shipModel, TR(px, py, deckZ(px, py)), RZ(yaw));
  for (const [part, m] of avatarPose(base, { phase: p.phase, walk: p.walk, wave: Math.min(p.wave, 1), hold, time, look: p.mode === 'walk' ? p.pitch : 0 })) draw(meshes[part], m);
}
function project(p: V3, vp: Mat4): [number, number] | null {
  const x = vp[0] * p[0] + vp[4] * p[1] + vp[8] * p[2] + vp[12], y = vp[1] * p[0] + vp[5] * p[1] + vp[9] * p[2] + vp[13], w = vp[3] * p[0] + vp[7] * p[1] + vp[11] * p[2] + vp[15];
  if (w < 0.1) return null;
  return [(x / w * 0.5 + 0.5) * innerWidth, (1 - (y / w * 0.5 + 0.5)) * innerHeight];
}

function render(shipModel: Mat4) {
  const aspect = canvas.width / canvas.height;
  cam.fov = fitFov(cam.fov, aspect);
  const view = lookAt(cam.eye, [cam.eye[0] + cam.fwd[0], cam.eye[1] + cam.fwd[1], cam.eye[2] + cam.fwd[2]], cam.up);
  const vp = multiply(perspective(cam.fov, aspect, cam.near, 9000), view);
  curVP = vp;
  G.bindFramebuffer(G.FRAMEBUFFER, null);
  G.viewport(0, 0, canvas.width, canvas.height);
  G.clearColor(0, 0, 0, 1); G.clear(G.COLOR_BUFFER_BIT | G.DEPTH_BUFFER_BIT);

  G.disable(G.DEPTH_TEST); G.depthMask(false);
  common(pSky.use()).v3('uFwd', cam.fwd).v3('uRight', cam.right).v3('uUp', cam.up).f('uTanHalf', Math.tan(cam.fov / 2)).f('uAspect', aspect)
    .v3('uSkyTop', atm.sky).v3('uIsland', [IC.x, IC.y, IC.R]).v2('uWindDir', windVec[0], windVec[1]).f('uStorm', smoothstep(0.1, 0.8, atm.s));
  G.bindVertexArray(quad.vao); G.drawArrays(G.TRIANGLE_STRIP, 0, 4);
  G.enable(G.DEPTH_TEST); G.depthMask(true);

  common(pSolid.use()).m4('uVP', vp);
  draw(land, IDENT, 1);
  BUOYS.forEach((b, i) => draw(buoyMeshes[b.kind], buoyModels[i]));
  draw(shipMesh, shipModel);
  for (const mt of MASTS) for (const y of mt.yards) draw(sailMesh, mul(shipModel, TR(mt.x + 0.75, 0, y.z), SC(y.w, y.w, y.h * ship.sailSet)));
  CANNONS.forEach((c, i) => draw(cannonMesh, mul(shipModel, TR(c.x, c.y, c.z), RZ(c.side > 0 ? Math.PI / 2 : -Math.PI / 2), TR(-recoil[i] * 0.8, 0, 0))));
  const harpooner = me.mode === 'harpoon' ? me : [...others.values()].find((o) => o.mode === 'harpoon');
  const hy = harpooner?.aimYaw ?? 0, hp = harpooner?.aimPitch ?? 0.05;
  const hm = mul(shipModel, TR(HARPOON.x, 0, HARPOON.z), RZ(hy), RYm(-hp));
  draw(harpoonMesh, hm);
  if (harpoonCd <= 0) draw(boltMesh, mul(hm, TR(0, 0, 0.35)));
  draw(wheelMesh, mul(shipModel, TR(HELM.x, 0, HELM.z), RXm(ship.rudder * 3), SC(2.0, 2.0, 2.0)));
  for (const o of others.values()) drawAvatar(o, shipModel);
  if (playing && camView === 'tp' && me.mode !== 'harpoon') drawAvatar(me, shipModel);
  // Serpents.
  for (const m of monsters.list) {
    for (let i = m.segs.length - 1; i >= 1; i--) {
      const p = m.segs[i], q = m.segs[i - 1], r = segRadius(i);
      const yaw = Math.atan2(q[1] - p[1], q[0] - p[0]), pitch = Math.atan2(q[2] - p[2], Math.hypot(q[0] - p[0], q[1] - p[1]));
      draw(segMesh, mul(TR(p[0], p[1], p[2]), RZ(yaw), RYm(-pitch), SC(r * 1.25, r, r)));
    }
    if (m.segs.length > 1) {
      const h = m.segs[0], n = m.segs[1];
      const yaw = Math.atan2(h[1] - n[1], h[0] - n[0]), pitch = Math.atan2(h[2] - n[2], Math.hypot(h[0] - n[0], h[1] - n[1]));
      draw(headMesh, mul(TR(h[0], h[1], h[2]), RZ(yaw), RYm(-pitch * 0.7 - m.rear * 0.25)));
    }
  }
  for (const s of shots) {
    if (s.kind === 'c') draw(ballMesh, TR(s.p[0], s.p[1], s.p[2]));
    else draw(boltMesh, mul(TR(s.p[0], s.p[1], s.p[2]), RZ(Math.atan2(s.v[1], s.v[0])), RYm(-Math.atan2(s.v[2], Math.hypot(s.v[0], s.v[1])))));
  }

  common(pOcean.use()).m4('uVP', vp).v2('uCenter', Math.round(cam.eye[0] / 4) * 4, Math.round(cam.eye[1] / 4) * 4)
    .v4a('uWA', uWA).v4a('uWB', uWB).v3('uIsland', [IC.x, IC.y, IC.R]).i('uWake', 0).i('uShore', 1)
    .v3('uWakeRect', [WAKE.ox, WAKE.oy, WAKE.size]).v3('uShoreRect', [IC.x + SHORE.min, IC.y + SHORE.min, SHORE.size])
    .v3('uSkyTop', atm.sky).v2('uWind', windVec[0] * 0.15, windVec[1] * 0.15);
  G.activeTexture(G.TEXTURE0); G.bindTexture(G.TEXTURE_2D, wakeA.tex);
  G.activeTexture(G.TEXTURE1); G.bindTexture(G.TEXTURE_2D, shoreTex);
  G.bindVertexArray(ocean.vao); G.drawElements(G.TRIANGLES, ocean.count, G.UNSIGNED_INT, 0);

  G.enable(G.BLEND); G.depthMask(false);
  G.blendFunc(G.SRC_ALPHA, G.ONE_MINUS_SRC_ALPHA);
  common(pSpray.use()).m4('uVP', vp).f('uPx', canvas.height / Math.tan(cam.fov / 2));
  G.bindVertexArray(sprayBuf.vao); G.bindBuffer(G.ARRAY_BUFFER, sprayBuf.buf); G.bufferData(G.ARRAY_BUFFER, sprayPos, G.DYNAMIC_DRAW);
  G.drawArrays(G.POINTS, 0, MAXP);
  const rainCount = Math.floor((isMobile ? 2500 : 6000) * atm.rain);
  if (rainCount > 0) {
    common(pRain.use()).m4('uVP', vp).v3('uCamVel', [ship.vx, ship.vy, 0])
      .v3('uRainVel', [windVec[0] * 10 * atm.wind, windVec[1] * 10 * atm.wind, -11]).f('uBox', 44).m4('uBoatInv', IDENT)
      .f('uMask', 0).f('uAmt', 0.4 + 0.6 * atm.rain);
    G.bindVertexArray(rainVao); G.drawArrays(G.LINES, 0, rainCount * 2);
  }

  G.blendFunc(G.ONE, G.ONE);
  const beamAng = time * 0.55;
  const bm = fromAxes([Math.cos(beamAng), Math.sin(beamAng), 0], [-Math.sin(beamAng), Math.cos(beamAng), 0], [0, 0, 1], [LIGHTHOUSE.x, LIGHTHOUSE.y, LIGHTHOUSE.lamp]);
  pBeam.use().m4('uVP', vp).m4('uModel', bm).v3('uCamPos', cam.eye).f('uFogDen', atm.fogDen).f('uI', 0.6 + 0.6 * (1 - atm.s));
  G.bindVertexArray(beam.vao); G.drawArrays(G.TRIANGLES, 0, beam.count);

  glowCount = 0;
  const lamp: V3 = [LIGHTHOUSE.x, LIGHTHOUSE.y, LIGHTHOUSE.lamp];
  const toCam = norm3([cam.eye[0] - lamp[0], cam.eye[1] - lamp[1], 0]);
  const align = Math.abs(Math.cos(beamAng) * toCam[0] + Math.sin(beamAng) * toCam[1]);
  const ld = Math.hypot(cam.eye[0] - lamp[0], cam.eye[1] - lamp[1]);
  const glare = Math.pow(align, 80);
  addGlow(lamp, 5 + ld * 0.004 + glare * (40 + ld * 0.05), [1, 0.85, 0.6], (1.2 + 5 * glare) * Math.exp(-ld * atm.fogDen * 0.45));
  BUOYS.forEach((b, i) => {
    if (((time + b.phase) % 3) >= 0.6) return;
    const d = Math.hypot(cam.eye[0] - buoyTops[i][0], cam.eye[1] - buoyTops[i][1]);
    addGlow(buoyTops[i], 1.2 + d * 0.006, BUOY_LIGHT[b.kind], 1.4 * Math.exp(-d * atm.fogDen * 0.5));
  });
  for (const y of [-2.2, 2.2]) addGlow(shipPoint(-18.4, y, 7.1), 1.2, [1, 0.75, 0.4], 0.9);
  for (const m of monsters.list) if (m.segs[0] && m.st !== MS.Dying) {
    const h = m.segs[0], hd = Math.atan2(h[1] - (m.segs[1]?.[1] ?? h[1]), h[0] - (m.segs[1]?.[0] ?? h[0]));
    for (const s of [1, -1]) addGlow([h[0] + Math.cos(hd) * 2.3 - Math.sin(hd) * s * 0.95, h[1] + Math.sin(hd) * 2.3 + Math.cos(hd) * s * 0.95, h[2] + 0.6], 0.9, [1, 0.85, 0.2], 0.9 + m.hurt);
  }
  if (glowCount) {
    pGlow.use().m4('uVP', vp).v3('uCamRight', cam.right).v3('uCamUp', cam.up);
    G.bindVertexArray(glowBuf.vao); G.bindBuffer(G.ARRAY_BUFFER, glowBuf.buf);
    G.bufferData(G.ARRAY_BUFFER, glowData.subarray(0, glowCount * 60), G.DYNAMIC_DRAW);
    G.drawArrays(G.TRIANGLES, 0, glowCount * 6);
  }
  if (bolt.t < 0.32 && bolt.pts.length) {
    const verts: number[] = [];
    bolt.pts.forEach((line, bi) => {
      for (let k = 0; k < line.length - 1; k++) {
        const p = line[k], q = line[k + 1];
        const mid: V3 = [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2, (p[2] + q[2]) / 2];
        const w = (bi === 0 ? 9 : 5) + Math.hypot(mid[0] - cam.eye[0], mid[1] - cam.eye[1]) * 0.003;
        const sd = norm3(cross(sub(q, p), sub(cam.eye, mid)));
        const a0: V3 = [p[0] - sd[0] * w, p[1] - sd[1] * w, p[2] - sd[2] * w], a1: V3 = [p[0] + sd[0] * w, p[1] + sd[1] * w, p[2] + sd[2] * w];
        const b0: V3 = [q[0] - sd[0] * w, q[1] - sd[1] * w, q[2] - sd[2] * w], b1: V3 = [q[0] + sd[0] * w, q[1] + sd[1] * w, q[2] + sd[2] * w];
        verts.push(...a0, 0, ...a1, 1, ...b1, 1, ...a0, 0, ...b1, 1, ...b0, 0);
      }
    });
    pBolt.use().m4('uVP', vp).f('uI', flash * 2.2);
    G.bindVertexArray(boltBuf.vao); G.bindBuffer(G.ARRAY_BUFFER, boltBuf.buf);
    G.bufferData(G.ARRAY_BUFFER, new Float32Array(verts), G.DYNAMIC_DRAW);
    G.drawArrays(G.TRIANGLES, 0, verts.length / 4);
  }
  G.disable(G.BLEND); G.depthMask(true);
  G.bindVertexArray(null);
}

function updateTags() {
  for (const o of others.values()) {
    if (!o.tag) { o.tag = document.createElement('div'); o.tag.className = 'tag'; tagsEl.appendChild(o.tag); }
    o.tag.textContent = o.name; o.tag.style.color = COAT_CSS[o.color % COAT_CSS.length];
    const head = shipPoint(o.x, o.y, deckZ(o.x, o.y) + 2.35);
    const s = project(head, curVP);
    const d = dist3(head, cam.eye);
    if (s && d < 120 && playing) { o.tag.style.display = 'block'; o.tag.style.left = `${s[0]}px`; o.tag.style.top = `${s[1]}px`; o.tag.style.opacity = String(clamp(1.4 - d / 80, 0.3, 1)); }
    else o.tag.style.display = 'none';
  }
}

// ---------- Networking ----------
interface Snap { t: number; sh: number[]; }
const shipSnaps: Snap[] = [];
let clockOffset: number | null = null;
const DELAY = 0.12;
const now = () => performance.now() / 1000;
let sendT = 0, stateT = 0;

function hostTimeNow() { return clockOffset === null ? time : now() + clockOffset - DELAY; }
function interpShip(t: number) {
  if (shipSnaps.length === 0) return;
  let a = shipSnaps[0], b = shipSnaps[shipSnaps.length - 1];
  for (let i = 0; i < shipSnaps.length - 1; i++) if (shipSnaps[i].t <= t && shipSnaps[i + 1].t >= t) { a = shipSnaps[i]; b = shipSnaps[i + 1]; break; }
  const f = b.t > a.t ? clamp((t - a.t) / (b.t - a.t), 0, 1.5) : 0;
  const out = a.sh.map((v, i) => (i === 2 ? angLerp(v, b.sh[i], f) : v + (b.sh[i] - v) * f));
  ship.unpack(out);
}

net.onChange = () => {
  if (net.role === 'host' && clockOffset !== null) { clockOffset = null; toast('You are now the captain of this ship', 3); }
  for (const id of [...others.keys()]) if (!net.peers.has(id)) { const o = others.get(id)!; o.tag?.remove(); others.delete(id); if (helmId === id) helmId = null; toast(`${o.name} left the ship`, 2.5); }
  updateCrewPanel();
};
net.onMsg = (m: NetMsg) => {
  const from = m.from as number;
  switch (m.t) {
    case 'p': {
      let o = others.get(from);
      if (!o) { o = newPlayer(from, String(m.n), Number(m.c)); others.set(from, o); toast(`${o.name} came aboard`, 2.5); updateCrewPanel(); }
      if (o.name !== m.n || o.color !== m.c) { o.name = String(m.n); o.color = Number(m.c); updateCrewPanel(); }
      o.snaps.push({ t: now(), x: m.x as number, y: m.y as number, yaw: m.yw as number, pitch: m.pt as number, ay: m.ay as number, ap: m.ap as number });
      if (o.snaps.length > 8) o.snaps.shift();
      o.mode = m.md as Mode; o.moving = !!m.mv;
      if (m.wv) o.wave = 1.6;
      break;
    }
    case 's': {
      if (net.role !== 'guest') break;
      const t = m.tm as number, off = t - now();
      clockOffset = clockOffset === null ? off : clockOffset + (off - clockOffset) * 0.05;
      shipSnaps.push({ t, sh: m.sh as number[] });
      while (shipSnaps.length > 12) shipSnaps.shift();
      const prevHelm = helmId;
      helmId = (m.hm as number | null) ?? null;
      if (helmId === me.id && me.mode !== 'helm') enterHelm();
      if (helmId !== me.id && me.mode === 'helm') { me.mode = 'walk'; me.x = HELM.stand[0] - 0.8; }
      if (prevHelm !== helmId) updateCrewPanel();
      monsterTargets = m.mo as number[][];
      score = m.sc as number;
      break;
    }
    case 'claim': if (isHost() && helmId === null) { helmId = from; updateCrewPanel(); } break;
    case 'release': if (isHost() && helmId === from) { helmId = null; updateCrewPanel(); } break;
    case 'in': if (isHost() && helmId === from) remoteHelm = { rudder: m.r as number, sail: m.s as number }; break;
    case 'fire': if (isHost()) hostFire(m.k as 'c' | 'h', m.i as number, m.ay as number, m.ap as number); break;
    case 'shot': if (!isHost()) spawnShot(m.id as number, m.k as 'c' | 'h', m.i as number, m.p as V3, m.v as V3); break;
    case 'hit': if (!isHost()) onHit(m.id as number, m.p as V3, m.m as number, !!m.k); break;
    case 'ram': if (!isHost()) rammed(m.p as V3); break;
    case 'wreck': if (!isHost()) showWreck(); break;
  }
};
let monsterTargets: number[][] | null = null;
function guestMonsters(dt: number) {
  if (!monsterTargets) return;
  const k = 1 - Math.exp(-dt * 10);
  const byId = new Map(monsters.list.map((x) => [x.id, x]));
  const rows = monsterTargets.map((r) => {
    const m = byId.get(r[0]);
    if (!m) return r;
    return [r[0], m.x + (r[1] - m.x) * k, m.y + (r[2] - m.y) * k, angLerp(m.hd, r[3], k), r[4], r[5], r[6], m.rear + (r[7] - m.rear) * k, r[8], r[9]];
  });
  monsters.apply(rows, time);
}
function interpOthers() {
  const t = now() - DELAY;
  for (const o of others.values()) {
    const s = o.snaps;
    if (!s.length) continue;
    let a = s[0], b = s[s.length - 1];
    for (let i = 0; i < s.length - 1; i++) if (s[i].t <= t && s[i + 1].t >= t) { a = s[i]; b = s[i + 1]; break; }
    const f = b.t > a.t ? clamp((t - a.t) / (b.t - a.t), 0, 1) : 1;
    const nx = a.x + (b.x - a.x) * f, ny = a.y + (b.y - a.y) * f;
    const moved = Math.hypot(nx - o.x, ny - o.y);
    o.x = nx; o.y = ny; o.yaw = angLerp(a.yaw, b.yaw, f); o.pitch = a.pitch + (b.pitch - a.pitch) * f;
    o.aimYaw = a.ay + (b.ay - a.ay) * f; o.aimPitch = a.ap + (b.ap - a.ap) * f;
    o.walk += ((o.moving ? 1 : 0) - o.walk) * 0.2;
    o.phase += moved * 2.4 / 3.2 * 3.2;
    if (o.moving && Math.floor(o.phase / Math.PI) !== o.lastStep) { o.lastStep = Math.floor(o.phase / Math.PI); audio.step(dist3(shipPoint(o.x, o.y, deckZ(o.x, o.y)), cam.eye)); }
  }
}
function netSend(dt: number) {
  if (!net.online) return;
  sendT += dt; stateT += dt;
  if (sendT > 1 / 15) {
    sendT = 0;
    net.send({ t: 'p', n: me.name, c: me.color, x: +me.x.toFixed(2), y: +me.y.toFixed(2), yw: +me.yaw.toFixed(3), pt: +me.pitch.toFixed(3), md: me.mode, mv: me.moving ? 1 : 0, wv: me.wave > 1.5 ? 1 : 0, ay: +me.aimYaw.toFixed(3), ap: +me.aimPitch.toFixed(3) });
    if (net.role === 'guest' && me.mode === 'helm') net.send({ t: 'in', r: +helmRudder.toFixed(3), s: +ship.sail.toFixed(3) });
  }
  if (net.role === 'host' && stateT > 1 / 20) {
    stateT = 0;
    net.send({ t: 's', tm: +time.toFixed(3), sh: ship.pack().map((v) => +v.toFixed(4)), hm: helmId, mo: monsters.pack(), sc: score });
  }
}

// ---------- Wreck ----------
let wreckT = 0;
function showWreck() { document.body.classList.add('wrecked'); setTimeout(() => document.body.classList.remove('wrecked'), 4500); }

// ---------- Game step ----------
function handleShipEvents() {
  for (const e of ship.events) {
    if (e.type === 'slam') {
      audio.slap(e.i);
      cam.shake = Math.max(cam.shake, 0.06 * e.i);
      wakeBlob = 0.5 * e.i;
      const n = Math.round(40 + 120 * e.i);
      for (let k = 0; k < n; k++) emitLocal([15 + R() * 3, (R() - 0.5) * 6, 3 + R()], [-(1 + R() * 6 * e.i), (R() - 0.5) * (6 + 10 * e.i), 4 + R() * (6 + 12 * e.i)], 1.2);
    } else if (e.type === 'green') {
      for (let k = 0; k < 120 * e.i; k++) emitLocal([8 + R() * 8, (R() - 0.5) * 7, 5 + R() * 2], [-(5 + R() * 6), (R() - 0.5) * 4, 2 + R() * 4], 1.2);
      cam.shake = Math.max(cam.shake, 0.05);
    } else { audio.crunch(e.i); cam.shake = Math.max(cam.shake, 0.3 * e.i); toast('Rocks!', 1.5); }
  }
  ship.events.length = 0;
}

function hostStep(dt: number) {
  // Helm: from me, from a remote helmsman, or nobody (rudder eases back, sails stay set).
  if (helmId === me.id) { ship.rudder = helmRudder; ship.sail = clamp(ship.sail + helmSailRate * 0.35 * dt, 0, 1); }
  else if (helmId !== null) { ship.rudder += clamp(remoteHelm.rudder - ship.rudder, -3 * dt, 3 * dt); ship.sail = clamp(remoteHelm.sail, 0, 1); }
  else ship.rudder *= Math.exp(-dt * 0.8);
  const n = Math.max(1, Math.ceil(dt / (1 / 60)));
  for (let i = 0; i < n; i++) ship.step(dt / n, windVec[0] * 14 * atm.wind, windVec[1] * 14 * atm.wind);
  handleShipEvents();
  monsters.update(dt, ship, stormAt(ship.x, ship.y), time, onRam);
  if (ship.hull <= 0 && wreckT <= 0) { wreckT = 4.5; showWreck(); net.send({ t: 'wreck' }); }
  if (wreckT > 0) { wreckT -= dt; if (wreckT <= 0) { spawnShip('shelter'); score = 0; } }
}

function guestStep(dt: number) {
  if (me.mode === 'helm') ship.sail = clamp(ship.sail + helmSailRate * 0.35 * dt, 0, 1);
  const sail = ship.sail;
  interpShip(hostTimeNow());
  if (me.mode === 'helm') ship.sail = sail;
  guestMonsters(dt);
}

function simulate(dt: number) {
  if (net.role === 'guest' && clockOffset !== null) time = hostTimeNow(); else time += dt;
  updatePhases(time);
  if (playing) updateControls(dt);
  if (isHost()) hostStep(dt); else guestStep(dt);
  const speed = Math.hypot(ship.vx, ship.vy);
  if (speed > 3 && ship.imm > -3) for (const sg of [1, -1]) if (R() < 0.6) emitLocal([14 + R() * 2, sg * 3.2, 0.8], [-1 - R(), sg * (2 + R() * 3), 1.5 + R() * 3], 0.8);
  updateShots(dt);
  interpOthers();
  updateAtmosphere(dt);
  updateLightning(dt);
  updateBuoys();
  updateSpray(dt);
  updateWake();
  updateCamera(dt);
  if (playing) checkGates();
  const ax = shipAxes(ship.psi, ship.th, ship.ph);
  return fromAxes(ax.X, ax.Y, ax.Z, [ship.x, ship.y, ship.z]);
}

let last = performance.now();
function frame(nowMs: number) {
  const dt = Math.min((nowMs - last) / 1000, 0.05);
  last = nowMs;
  resize();
  const model = simulate(dt);
  render(model);
  updateTags();
  updateHud(dt);
  netSend(dt);
  audio.update({ speed: Math.min(Math.hypot(ship.vx, ship.vy) / 8, 1), storm: 0.15 + 0.85 * atm.s, rain: atm.rain, wind: atm.wind, heave: Math.abs(ship.vz), roll: Math.abs(ship.phv) }, dt);
  requestAnimationFrame(frame);
}

// ---------- Menu ----------
const PIRATE = ['Salty', 'One-Eyed', 'Red', 'Black', 'Mad', 'Calico', 'Iron', 'Grim', 'Bonny', 'Silver'];
const NAMES = ['Anne', 'Jack', 'Morgan', 'Ned', 'Mary', 'Flint', 'Rackham', 'Grace', 'Teach', 'Kidd'];
const nameEl = $<HTMLInputElement>('name');
let savedName = '';
try { savedName = localStorage.getItem('seas-name') ?? ''; } catch { /* storage blocked */ }
nameEl.value = savedName || `${PIRATE[Math.floor(R() * PIRATE.length)]} ${NAMES[Math.floor(R() * NAMES.length)]}`;
const joinCode = new URLSearchParams(location.search).get('crew')?.toUpperCase().replace(/[^A-Z0-9]/g, '') ?? '';
if (joinCode) document.body.classList.add('joining');
const setSpawn = (m: 'shelter' | 'storm') => { spawnMode = m; $('spShelter').classList.toggle('on', m === 'shelter'); $('spStorm').classList.toggle('on', m === 'storm'); };
$('spShelter').addEventListener('click', () => setSpawn('shelter'));
$('spStorm').addEventListener('click', () => setSpawn('storm'));
const statusEl = $('status');

function begin() {
  me.name = (nameEl.value.trim() || 'Pirate').slice(0, 18);
  try { localStorage.setItem('seas-name', me.name); } catch { /* storage blocked */ }
  playing = true;
  document.body.classList.add('playing');
  audio.start();
  me.id = net.id; me.color = net.role === 'guest' ? 1 + (net.id % 3) : 0;
  me.x = net.role === 'guest' ? -5 : -7; me.y = net.role === 'guest' ? 1.6 : -1.2; me.yaw = 0; me.pitch = 0;
  updateCrewPanel();
  if (!isTouch) lockPointer();
}
$('soloBtn').addEventListener('click', () => { spawnShip(spawnMode); begin(); toast(spawnMode === 'storm' ? 'High seas' : 'Sheltered water', 3); });
$('hostBtn').addEventListener('click', async () => {
  statusEl.textContent = 'Opening a crew channel…';
  try {
    await net.connect(Net.newRoom());
    spawnShip(spawnMode); begin();
    history.replaceState(null, '', `?crew=${net.room}`);
    void navigator.clipboard?.writeText(inviteLink()).catch(() => undefined);
    toast('Invite link copied: send it to your mate', 4);
  } catch (e) { statusEl.textContent = (e as Error).message + '. Try sailing solo.'; }
});
$('joinBtn').addEventListener('click', async () => {
  statusEl.textContent = 'Rowing out to the ship…';
  try {
    await net.connect(joinCode);
    if (net.role === 'host') { spawnShip(spawnMode); toast('No crew here yet: you are the captain', 3.5); }
    begin();
  } catch (e) { statusEl.textContent = (e as Error).message; }
});
$('leaveJoin').addEventListener('click', () => { history.replaceState(null, '', location.pathname); document.body.classList.remove('joining'); });
$('camBtn').addEventListener('click', toggleCam);
$('muteBtn').addEventListener('click', toggleMute);
$('fsBtn').addEventListener('click', () => {
  if (document.fullscreenElement) void document.exitFullscreen();
  else void document.documentElement.requestFullscreen?.().catch(() => undefined);
});
document.addEventListener('visibilitychange', () => { if (!document.hidden) last = performance.now(); });

spawnShip('shelter');
updateAtmosphere(10);
requestAnimationFrame(frame);

// Dev hooks: deterministic stepping while the tab is throttled, and state inspection.
const dev = window as unknown as Record<string, unknown>;
dev.__advance = (n: number, dt = 1 / 60) => {
  let model = simulate(dt);
  for (let i = 1; i < n; i++) model = simulate(dt);
  render(model); updateTags(); updateHud(0); netSend(dt);
};
dev.__game = {
  ship, me, others, monsters, keys, cam, atm, net, solo: () => $('soloBtn').click(), interact, fireAction, toggleCam, wave,
  spawn: (m: 'shelter' | 'storm') => spawnShip(m), look: (dx: number, dy: number) => { look.dx += dx; look.dy += dy; },
  shots, score: () => score, setHelm: (id: number | null) => { helmId = id; },
};

